from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.util.cuda import DeviceBuffer
from polygraphy.util import misc

from collections import OrderedDict
import contextlib
import os

import tensorrt as trt


def Calibrator(data_loader, cache=None, BaseClass=trt.IInt8MinMaxCalibrator,
               batch_size=None):
    """
    Supplies calibration data to TensorRT to calibrate the network for INT8 inference.

    Args:
        data_loader (Generator -> OrderedDict[str, np.ndarray]):
            A generator or iterable that yields a dictionary that maps input names to input NumPy buffers.

            In case you don't know details about the inputs ahead of time, you can access the
            `input_metadata` property in your data loader, which will be set to an `TensorMetadata` instance.
            Note that this does not work for generators or lists.

            The number of calibration batches is controlled by the number of items supplied
            by the data loader.


        cache (Union[str, file-like]):
                Path or file-like object to save/load the calibration cache.
                By default, the calibration cache is not saved.
        BaseClass (type): The type of calibrator to inherit from. Defaults to trt.IInt8MinMaxCalibrator.
        batch_size (int):
                [DEPRECATED] The size of each batch provided by the data loader.
    """
    class CalibratorClass(BaseClass):
        """
        Calibrator that supplies calibration data to TensorRT to calibrate the network for INT8 inference.
        """
        def __init__(self):
            # Must explicitly initialize parent for any trampoline class! Will mysteriously segfault without this.
            BaseClass.__init__(self)

            self.data_loader = data_loader
            self.cache = cache
            self.device_buffers = OrderedDict()
            self.reset()
            G_LOGGER.verbose("Created calibrator [cache={:}]".format(self.cache))

            self.cache_contents = None
            self.cache_tried_already = False
            self.batch_size = misc.default_value(batch_size, 1)


        def reset(self, input_metadata=None):
            """
            Reset this calibrator by attempting to rewind the data loader - note that
            this doesn't work for generators.

            Args:
                input_metadata (TensorMetadata):
                        Mapping of input names to their data types and shapes.
                        Passed along to the data loader if provided. Generally should not be required
                        unless using Polygraphy's included DataLoader for this calibrator.
            """
            if input_metadata is not None:
                with contextlib.suppress(AttributeError):
                    self.data_loader.input_metadata = input_metadata

            # Attempt to reset data loader
            self.data_loader_iter = iter(self.data_loader)
            self.num_batches = 0


        def get_batch_size(self):
            return self.batch_size


        def get_batch(self, names):
            try:
                host_buffers = next(self.data_loader_iter)
            except StopIteration:
                if not self.num_batches:
                    G_LOGGER.warning("Calibrator data loader provided no data. Possibilities include: (1) data loader "
                                     "has no data to provide, (2) data loader was a generator, and the calibrator is being "
                                     "reused across multiple loaders (generators cannot be rewound)")
                return None
            else:
                self.num_batches += 1

            for name, host_buffer in host_buffers.items():
                if name not in self.device_buffers:
                    self.device_buffers[name] = DeviceBuffer(shape=host_buffer.shape, dtype=host_buffer.dtype)
                    G_LOGGER.verbose("Allocated: {:}".format(self.device_buffers[name]))
                    if self.num_batches > 1:
                        G_LOGGER.warning("The calibrator data loader provided an extra input ({:}) compared to the last set of inputs.\n"
                                         "Should this input be removed, or did you accidentally omit an input before?".format(name))

                device_buffer = self.device_buffers[name]
                device_buffer.copy_from(host_buffer)
            return [device_buffer.address() for device_buffer in self.device_buffers.values()]


        def read_calibration_cache(self):
            if self.cache is None:
                return None

            def load_from_cache():
                try:
                    return self.cache.read()
                except AttributeError:
                    if os.path.exists(self.cache):
                        G_LOGGER.info("Reading calibration cache from: {:}".format(self.cache), mode=LogMode.ONCE)
                        with open(self.cache, "rb") as f:
                            return f.read()
                except:
                    # Cache is not readable
                    return None


            if not self.cache_tried_already:
                self.cache_tried_already = True
                self.cache_contents = load_from_cache()
                if self.cache_contents is not None and not self.cache_contents:
                    G_LOGGER.warning("Calibration cache was provided, but is empty. Will regenerate scales by running calibration.", mode=LogMode.ONCE)
                    self.cache_contents = None
            return self.cache_contents


        def write_calibration_cache(self, cache):
            if self.cache is None:
                return

            cache = cache.tobytes()
            try:
                bytes_written = self.cache.write(cache)
                if bytes_written != len(cache):
                    G_LOGGER.warning("Could not write entire cache. Note: cache contains {:} bytes, but only "
                                        "{:} bytes were written".format(len(cache), bytes_written))
            except AttributeError:
                G_LOGGER.info("Writing calibration cache to: {:}".format(self.cache))
                with open(self.cache, "wb") as f:
                    f.write(cache)
            except:
                # Cache is not writable
                return
            else:
                self.cache.flush()


        def __del__(self):
            for device_buffer in self.device_buffers.values():
                device_buffer.free()


    return CalibratorClass()
