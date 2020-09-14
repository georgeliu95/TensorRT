from polygraphy.comparator.data_loader import DataLoader, DataLoaderCache
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.common import constants, TensorMetadata
from polygraphy.util import misc

from collections import OrderedDict, defaultdict
from multiprocessing import Queue, Process
import contextlib
import queue
import time
import copy
import sys

import numpy as np



class IterationResult(OrderedDict):
    def __init__(self, outputs=None, runtime=None, runner_name=None):
        """
        An ordered dictionary containing the result of a running a single iteration of a runner.

        This maps output names to NumPy arrays, and preserves the output ordering from the runner.

        Also includes additional fields indicating the name of the runner which produced the
        outputs, and the time required to do so.


        Args:
            outputs (Dict[str, np.array]): The outputs of this iteration, mapped to their names.


            runtime (float): The time required for this iteration, in seconds.
            runner_name (str): The name of the runner that produced this output.
        """
        # IMPORTANT: This class must be pickleable.
        initial = misc.default_value(outputs, {})
        # Before 3.6, OrderedDict.update() did not preserve ordering
        for key, val in initial.items():
            self[key] = val
        self.runtime = runtime
        self.runner_name = misc.default_value(runner_name, "")


class RunResults(OrderedDict):
    def __init__(self):
        """
        An ordered dictionary that maps runners, by name, to zero or more IterationResults.

        More specifically, it is an OrderedDict[str, List[IterationResult]].
        """
        pass


class AccuracyResult(OrderedDict):
    """
    An ordered dictionary including details about the result of `Comparator.compare_accuracy`.

    More specifically, it is an OrderedDict[Tuple[str, str], List[OrderedDict[str, bool]]] which maps a runner
    pair (a tuple containing both runner names) to a list of dictionaries of booleans (or anything that can be
    converted into a boolean, such as an OutputCompareResult), indicating whether there was a match in the outputs of
    the corresponding iteration. The List[OrderedDict[str, bool]] is constructed from the dictionaries returned
    by `compare_func` in `compare_accuracy`.

    For example, to see if there was a match in "output0" between "runner0" and
    "runner1" on the 1st iteration, you would do the following:
    ::
        output_name = "output0"
        runner_pair = ("runner0", "runner1")
        iteration = 0
        match = bool(accuracy_result[runner_pair][iteration][output_name])

    In case there's a mismatch, you would be able to inspect the outputs in question by accessing
    the results from Comparator.run() (assumed here to be called run_results):
    ::
        runner0_output = run_results["runner0"][iteration][output_name]
        runner1_output = run_results["runner1"][iteration][output_name]

    Note: This class overrides __bool__(), and returns True only if all outputs matched for every iteration.
    Thus, you can avoid manually checking each output, and instead write something like:
    ::
        if accuracy_result:
            print("All matched!")
    """
    def __bool__(self):
        return all([bool(match) for outs in self.values() for out in outs for match in out.values()])


    def _get_runner_pair(self, runner_pair):
        return misc.default_value(runner_pair, list(self.keys())[0])


    def percentage(self, runner_pair=None):
        """
        Returns the percentage of iterations that matched for the given pair of runners,
        expressed as a decimal between 0.0 and 1.0.

        Always returns 1.0 when the number of iterations is 0, or when there are no runner comparisons.

        Args:
            runner_pair ((str, str)):
                    A pair of runner names describing which runners to check.
                    Defaults to the first pair in the dictionary.
        """
        if not list(self.keys()):
            return 1.0 # No data in this result.

        matched, _, total = self.stats(runner_pair)
        if not total:
            return 1.0 # No iterations
        return float(matched) / float(total)


    def stats(self, runner_pair=None):
        """
        Returns the number of iterations that matched, mismatched, and the total number of iterations.

        Args:
            runner_pair ((str, str)):
                    A pair of runner names describing which runners to check.
                    Defaults to the first pair in the dictionary.

        Returns:
            (int, int, int): Number of iterations that matched, mismatched, and total respectively.
        """
        runner_pair = self._get_runner_pair(runner_pair)
        outs = self[runner_pair]
        matched = sum([all([match for match in out.values()]) for out in outs])
        total = len(outs)
        return matched, total - matched, total


# Provides functions to post-process outputs.
class PostprocessFunc(object):
    @staticmethod
    # This function returns a top_k function that can be used as a postprocess_func.
    def topk_func(k=10, axis=1, exclude_outputs=set()):
        """
        Creates a function that applies a Top-K operation to a IterationResult.
        Top-K will return the indices of the k largest values in the array.

        Args:
            k (int): The number of indices to keep. Defaults to 10.
            axis (int): The axis along which to apply the topk. Defaults to 1.
            exclude_outputs (Set[str]):
                    Names of outputs to exclude. Top-K will not be applied to these outputs.

        Returns:
            Callable(IterationResult) -> IterationResult: The top-k function.
        """
        # Top-K implementation.
        def topk(run_result):
            for name, output in run_result.items():
                if name not in exclude_outputs:
                    indices = np.argsort(-output, axis=axis)
                    run_result[name] = np.take(indices, np.arange(0, k), axis=axis)
            return run_result
        return topk


class OutputCompareResult(object):
    def __init__(self, passed, required_atol, required_rtol):
        """
        Represents the result of comparing a single output of a single iteration
        between two runners.

        Args:
            passed (bool): Whether the error was within acceptable limits.
            required_atol (float): The minimum required absolute tolerance to consider the outputs equivalent.
            required_rtol (float): The minimum required relative tolerance to consider the outputs equivalent.
        """
        self.passed = passed
        self.required_atol = required_atol
        self.required_rtol = required_rtol

    def __bool__(self):
        return self.passed


    def __str__(self):
        return "(atol={:}, rtol={:})".format(self.required_atol, self.required_rtol)


# Provides functions to compare two IterationResults
class CompareFunc(object):
    @staticmethod
    def basic_compare_func(check_shapes=None, rtol=None, atol=None, fail_fast=None, find_output_func=None):
        """
        Creates a function that compares two IterationResults, and can be used as the `compare_func` argument
        in ``Comparator.compare_accuracy``.
        This function uses ``np.isclose`` to determine whether outputs match.
        For details, see https://docs.scipy.org/doc/numpy/reference/generated/numpy.isclose.html.

        Args:
            check_shapes (bool):
                    Whether shapes must match exactly. If this is False, this function may
                    permute or reshape outputs before comparison. Defaults to True.
            rtol (float):
                    The relative tolerance to use when checking accuracy. Defaults to 1e-5.
            atol (float):
                    The absolute tolerance to use when checking accuracy. Defaults to 1e-5.
            fail_fast (bool):
                    Whether the function should exit immediately after the first failure. Defaults to False.
            find_output_func (Callable(str, int, IterationResult) -> List[str]):
                    A callback that returns a list of output names to compare against from the provided
                    IterationResult, given an output name and index from another IterationResult.
                    The comparison function will always iterate over the output names of the
                    first IterationResult, expecting names from the second. A return value of
                    `[]` or `None` indicates that the output should be skipped.

        Returns:
            Callable(IterationResult, IterationResult) -> OrderedDict[str, bool]
        """
        check_shapes = misc.default_value(check_shapes, True)
        rtol = misc.default_value(rtol, 1e-5)
        atol = misc.default_value(atol, 1e-5)
        fail_fast = misc.default_value(fail_fast, False)


        def compare_output(run_result0, run_result1):
            """
            Compare the outputs of two runners from a single iteration.

            This function will always iterate over the output names of the first IterationResult,
                and attempt to find corresponding output names in the second.
            If no corresponding output name is found, the output is skipped.
            If all output names are skipped, then this function raises an error.

            Args:
                run_result0 (IterationResult): The result of the first runner.
                run_result1 (IterationResult): The result of the second runner.

            Returns:
                OrderedDict[str, OutputCompareResult]:
                        The name of the outputs compared, derived from the first IterationResult,
                        and whether they matched. If an output name is not found, it is omitted from this dictionary.

            Raises:
                PolygraphyException: If all output names are skipped, and thus no outputs are compared.
            """
            # Returns whether the outputs match
            def check_outputs_match(out0, out0_name, out1, out1_name):
                def compute_amax(buffer):
                    if not misc.is_empty_shape(buffer.shape):
                        return np.amax(buffer)
                    return 0

                def compute_required():
                    # The purpose of this function is to determine the minimum tolerances such that
                    # the outputs would be considered a match.
                    # The NumPy formula for np.isclose is absolute(out0 - out1) <= (atol + rtol * absolute(out1))
                    # So, for both absolute/relative tolerance, given either one,
                    # we can compute the required value for the other:
                    # atol = absolute(out0 - out1)
                    # atol_if_rtol = absolute(out0 - out1)  - rtol * absolute(out1)
                    # rtol = (absolute(out0 - out1) - atol) / absolute(out1)
                    try:
                        absdiff = np.abs(out0 - out1)
                        absout1 = np.abs(out1)
                        required_atol = max(compute_amax(absdiff), 0.0)
                        required_atol_if_rtol = max(compute_amax(absdiff - rtol * absout1), 0.0)
                        required_rtol = max(compute_amax((absdiff - atol) / absout1), 0.0)
                        return required_atol, required_atol_if_rtol, required_rtol
                    except:
                        return None, None, None

                def log_mismatches(mismatches):
                    try:
                        with G_LOGGER.indent():
                            G_LOGGER.error("Note: Use -vvv or set logging verbosity to SUPER_VERBOSE to display mismatches", mode=LogMode.ONCE)
                            G_LOGGER.super_verbose("Mismatches at:\n" + str(mismatches))
                            G_LOGGER.extra_verbose("Runner: {:40} | Mismatched values:\n{:}".format(run_result0.runner_name, out0[mismatches]))
                            G_LOGGER.extra_verbose("Runner: {:40} | Mismatched values:\n{:}".format(run_result1.runner_name, out1[mismatches]))
                    except:
                        G_LOGGER.warning("Failing to log mismatches - this may be because the outputs are of different shapes")


                try:
                    mismatches = np.logical_not(np.isclose(output0, output1, rtol=rtol, atol=atol))
                except Exception as err:
                    G_LOGGER.warning("Failed to compare outputs with:\n{:}\nSkipping".format(err))
                    return False

                required_atol, required_atol_if_rtol, required_rtol = compute_required()
                log_msg = "Minimum required tolerances: [atol={:}] OR [rtol={:}, atol={:}] OR [rtol={:}, atol={:}]\n".format(
                                required_atol, rtol, required_atol_if_rtol, required_rtol, atol)
                G_LOGGER.super_verbose("Runner: {:40} | Output (dtype={:}, shape={:}):\n{:}".format(run_result0.runner_name, out0.dtype, out0.shape, out0))
                G_LOGGER.super_verbose("Runner: {:40} | Output (dtype={:}, shape={:}):\n{:}".format(run_result1.runner_name, out1.dtype, out1.shape, out1))

                failed = np.any(mismatches)
                if failed:
                    log_mismatches(mismatches)
                    G_LOGGER.error(log_msg)
                    G_LOGGER.error("FAILED | Difference exceeds tolerance (rtol={:}, atol={:})".format(rtol, atol))
                else:
                    G_LOGGER.info(log_msg)
                    G_LOGGER.success("PASSED | Difference is within tolerance (rtol={:}, atol={:})".format(rtol, atol))

                G_LOGGER.verbose("Finished comparing: '{:}' (dtype={:}, shape={:}) [{:}] and '{:}' (dtype={:}, shape={:}) [{:}]"
                                .format(out0_name, out0.dtype, out0.shape, run_result0.runner_name, out1_name, out1.dtype, out1.shape, run_result1.runner_name))
                return OutputCompareResult(not failed, required_atol, required_rtol)


            output_status = OrderedDict() # OrderedDict[str, bool] Maps output names to whether they matched.

            if not check_shapes:
                G_LOGGER.info("Strict shape checking disabled. Will attempt to match output shapes before comparisons")


            def default_find_output_func(output_name, index, run_result):
                found_name = misc.find_in_dict(output_name, run_result, index)
                if found_name is None:
                    return None
                elif found_name != output_name:
                    exact_match = misc.find_in_dict(found_name, run_result0)
                    if exact_match == found_name:
                        G_LOGGER.verbose("Will not compare {:} with {:}, since the former already has an exact match: {:}".format(
                                            found_name, output_name, exact_match))
                        return None # If the found output is being compared against another output already, skip this non-exact match
                    G_LOGGER.warning("Output names did not match exactly. Assuming {:} output: {:} "
                                    "corresponds to output: {:}".format(
                                        run_result.runner_name, found_name, output_name))
                return [found_name]


            nonlocal find_output_func
            find_output_func = misc.default_value(find_output_func, default_find_output_func)

            for index, (out0_name, output0) in enumerate(run_result0.items()):
                out1_names = misc.default_value(find_output_func(out0_name, index, run_result1), [])

                if len(out1_names) > 1:
                    G_LOGGER.info("Will attempt to compare output: '{:}' [{:}] with multiple outputs: '{:}' [{:}]".format(
                                    out0_name, run_result0.runner_name, list(out1_names), run_result1.runner_name))

                for out1_name in out1_names:
                    if out1_name is None or out1_name not in run_result1:
                        G_LOGGER.warning("For output: '{:}' [{:}], skipping corresponding output: '{:}' [{:}], "
                                         "since the output was not found".format(out0_name, run_result0.runner_name,
                                                                                 out1_name, run_result1.runner_name))
                        continue

                    output1 = run_result1[out1_name]
                    G_LOGGER.info("Comparing Output: '{:}' (dtype={:}, shape={:}) with '{:}' (dtype={:}, shape={:})".format(
                                        out0_name, output0.dtype, output0.shape, out1_name, output1.dtype, output1.shape))
                    G_LOGGER.verbose("Note: Comparing {:} vs. {:}".format(run_result0.runner_name, run_result1.runner_name))

                    with G_LOGGER.indent():
                        if check_shapes and output0.shape != output1.shape:
                            G_LOGGER.error("Will not compare outputs of different shapes. Note: Output shapes are "
                                           "{:} and {:}.".format(output0.shape, output1.shape))
                            G_LOGGER.error("Note: Use --no-strict-shape-checking or set check_shapes=False to "
                                           "attempt to compare values anyway.", mode=LogMode.ONCE)
                            outputs_match = False
                        else:
                            output1 = misc.try_match_shape(output1, output0.shape)
                            output0 = output0.reshape(output1.shape)
                            outputs_match = check_outputs_match(output0, out0_name, output1, out1_name)

                        output_status[out0_name] = outputs_match
                        if fail_fast and not outputs_match:
                            return output_status

            mismatched_output_names = [name for name, matched in output_status.items() if not matched]
            if mismatched_output_names:
                G_LOGGER.error("Mismatched outputs: {:}".format(mismatched_output_names))

            # This is useful for catching cases were Polygraphy does something wrong with the runner output buffers
            if not output_status and (bool(run_result0.keys()) or bool(run_result1.keys())):
                r0_name = run_result0.runner_name
                r0_outs = list(run_result0.keys())
                r1_name = run_result1.runner_name
                r1_outs = list(run_result1.keys())
                G_LOGGER.critical("All outputs were skipped, no common outputs found! Note:\n{:} outputs: "
                                  "{:}\n{:} outputs: {:}".format(r0_name, r0_outs, r1_name, r1_outs))

            return output_status

        return compare_output


class Comparator(object):
    @staticmethod
    def run(runners, data_loader=None, warm_up=None,
            use_subprocess=None, subprocess_timeout=None,
            subprocess_polling_interval=None):
        """
        Runs the supplied runners sequentially.

        Args:
            data_loader (Generator -> OrderedDict[str, np.ndarray]):
                    A generator or iterable that yields a dictionary that maps input names to input numpy buffers.
                    In the simplest case, this can be a `list` of `dict`s.

                    In case you don't know details about the inputs ahead of time, you can access the
                    `input_metadata` property in your data loader, which will be set to an `TensorMetadata`
                    instance by this function.
                    Note that this does not work for generators or lists.

                    The number of iterations run by this function is controlled by the number of items supplied
                    by the data loader.

                    Defaults to an instance of `DataLoader`.
            warm_up (int):
                    The number of warm up runs to perform for each runner before timing.
                    Defaults to 0.
            use_subprocess (bool):
                    Whether each runner should be run in a subprocess. This allows each runner to have exclusive
                    access to the GPU. When using a subprocess, runners and loaders will never be modified.
            subprocess_timeout (int):
                    The timeout before a subprocess is killed automatically. This is useful for handling processes
                    that never terminate. A value of None disables the timeout. Defaults to None.
            subprocess_polling_interval (int):
                    The polling interval, in seconds, for checking whether a subprocess has completed or crashed.
                    In rare cases, omitting this parameter when subprocesses are enabled may cause this function
                    to hang indefinitely if the subprocess crashes.
                    A value of 0 disables polling. Defaults to 30 seconds.

        Returns:
            RunResults: A mapping of runner names to the results of their inference. The ordering of `runners` is preserved in this mapping.
        """
        warm_up = misc.default_value(warm_up, 0)
        data_loader = misc.default_value(data_loader, DataLoader())
        use_subprocess = misc.default_value(use_subprocess, False)
        subprocess_polling_interval = misc.default_value(subprocess_polling_interval, 30)
        loader_cache = DataLoaderCache(data_loader)


        def execute_runner(runner, loader_cache):
            with runner as active_runner:
                input_metadata = active_runner.get_input_metadata()
                G_LOGGER.verbose("Runner: {:40} | Input Metadata:\n{:}".format(active_runner.name, input_metadata))
                loader_cache.set_input_metadata(input_metadata)

                if warm_up:
                    G_LOGGER.info("Runner: {:40} | Running {:} warm-up runs".format(active_runner.name, warm_up))
                    try:
                        feed_dict = loader_cache[0]
                    except IndexError:
                        G_LOGGER.warning("{:} warm-up runs were requested, but data loader did not supply any data. "
                                         "Skipping warm-up runs".format(warm_up))
                    else:
                        G_LOGGER.ultra_verbose("Warm-up Input Buffers: {:}".format(feed_dict))
                        # First do a few warm-up runs, and don't time them.
                        for i in range(warm_up):
                            active_runner.infer(feed_dict=feed_dict)

                # Then, actual iterations.
                total_time = 0
                run_results = []
                for feed_dict in loader_cache:
                    G_LOGGER.extra_verbose(lambda: "Runner: {:40} | Feeding inputs:\n{:}".format(active_runner.name, feed_dict))
                    outputs = active_runner.infer(feed_dict=feed_dict)

                    runtime = active_runner.last_inference_time()
                    # Without a deep copy here, outputs will always reference the output of the last run
                    run_results.append(IterationResult(outputs=copy.deepcopy(outputs), runtime=runtime, runner_name=active_runner.name))

                    if len(run_results) == 1:
                        output_metadata = TensorMetadata()
                        for name, out in outputs.items():
                            output_metadata.add(name, out.dtype, out.shape)

                    G_LOGGER.verbose("Runner: {:40} | Output Metadata:\n{:}".format(active_runner.name, output_metadata), mode=LogMode.ONCE)
                    G_LOGGER.extra_verbose(lambda: "Runner: {:40} | Inference Time: {:.3f} ms | Received outputs:\n{:}".format(active_runner.name, runtime * 1000.0, outputs))

                G_LOGGER.info("Runner: {:40} | Completed {:} iterations.".format(active_runner.name, len(run_results)))
                return run_results



        # Wraps execute_runner to use a queue.
        def execute_runner_with_queue(runner_queue, runner, loader_cache):
            run_results = None
            try:
                run_results = execute_runner(runner, loader_cache)
            except:
                # Cannot send the exception back, as it is not necessarily pickleable
                import traceback
                G_LOGGER.error(traceback.format_exc())
            misc.try_send_on_queue(runner_queue, run_results)
            # After finishing, send the updated loader_cache back.
            misc.try_send_on_queue(runner_queue, loader_cache)


        # Do all inferences in one loop, then comparisons at a later stage.
        # We run each runner in a separate process so that we can provide exclusive GPU access for each runner.
        runner_queue = Queue()
        run_results = RunResults()
        for index, runner in enumerate(runners):
            G_LOGGER.info("Runner: {:40} | Activating and starting inference".format(runner.name))
            if use_subprocess:
                process = Process(target=execute_runner_with_queue, args=(runner_queue, runner, loader_cache))
                process.start()

                # If a subprocess hangs in a certain way, then process.join could block forever. Hence,
                # we need to keep polling the process to make sure it really is alive.
                run_results[runner.name] = None
                while process.is_alive() and run_results[runner.name] is None:
                    try:
                        run_results[runner.name] = misc.try_receive_on_queue(runner_queue, timeout=subprocess_polling_interval / 2)
                        # Receive updated loader cache, or fall back if it could not be sent.
                        loader_cache = misc.try_receive_on_queue(runner_queue, timeout=subprocess_polling_interval / 2)
                    except queue.Empty:
                        G_LOGGER.extra_verbose("Polled subprocess - still running")

                try:
                    assert run_results[runner.name] is not None
                    process.join(subprocess_timeout)
                except:
                    G_LOGGER.critical("Runner: {:40} | Terminated prematurely. Check the exception logged above. "
                                      "If there is no exception logged above, make sure not to use the --use-subprocess "
                                      "flag or set use_subprocess=False in Comparator.run().".format(runner.name))
                finally:
                    process.terminate()

                if loader_cache is None:
                    G_LOGGER.critical("Could not send data loader cache to runner subprocess. Please try disabling subprocesses "
                                      "by removing the --use-subprocess flag, or setting use_subprocess=False in Comparator.run()")
            else:
                run_results[runner.name] = execute_runner(runner, loader_cache)

        G_LOGGER.info("Successfully ran: {:}".format([r.name for r in runners]))
        return run_results


    @staticmethod
    def postprocess(run_results, postprocess_func):
        """
        Applies post processing to all the outputs in the provided run results.
        This is a convenience function to avoid the need for manual iteration over the run_results dictionary.

        Args:
            run_results (RunResults): The result of Comparator.run().
            postprocess_func (Callable(IterationResult) -> IterationResult):
                    The function to apply to each ``IterationResult``.

        Returns:
            RunResults: The updated run results.
        """
        for runner_name, results in run_results.items():
            for index, result in enumerate(results):
                results[index] = postprocess_func(result)
        return run_results


    @staticmethod
    def default_comparisons(run_results):
        # Sets up default comparisons - which is to compare each runner to the subsequent one.
        all_runners = list(run_results.keys())
        return [(r1, r2) for r1, r2 in zip(all_runners[:-1], all_runners[1:])]


    @staticmethod
    def compare_accuracy(run_results, fail_fast=False, comparisons=None, compare_func=None):
        """
        Args:
            run_results (RunResults): The result of Comparator.run()


            fail_fast (bool): Whether to exit after the first failure
            comparisons (List[Tuple[str, str]]):
                    Comparisons to perform, specified by runner names. For example, [(r0, r1), (r1, r2)]
                    would compare the runner named r0 with r1, and r1 with r2.
                    By default, this compares each result to the subsequent one.
            compare_func (Callable(IterationResult, IterationResult) -> OrderedDict[str, bool]):
                    A function that takes in two IterationResults, and returns a dictionary that maps output
                    names to a boolean (or anything convertible to a boolean) indicating whether outputs matched.
                    The order of arguments to this function is guaranteed to be the same as the ordering of the
                    tuples contained in `comparisons`.

        Returns:
            AccuracyResult:
                    A summary of the results of the comparisons. The order of the keys (i.e. runner pairs) is
                    guaranteed to be the same as the order of `comparisons`. For more details, see the AccuracyResult
                    docstring (e.g. help(AccuracyResult)).
        """
        def find_mismatched(match_dict):
            return [name for name, matched in match_dict.items() if not bool(matched)]

        compare_func = misc.default_value(compare_func, CompareFunc.basic_compare_func())
        comparisons = misc.default_value(comparisons, Comparator.default_comparisons(run_results))

        accuracy_result = AccuracyResult()
        for runner0_name, runner1_name in comparisons:
            G_LOGGER.info("Accuracy Comparison | {:} vs. {:}".format(runner0_name, runner1_name))
            with G_LOGGER.indent():
                results0, results1 = run_results[runner0_name], run_results[runner1_name]
                runner_pair = (runner0_name, runner1_name)
                accuracy_result[runner_pair] = []

                num_iters = min(len(results0), len(results1))
                for iteration, (result0, result1) in enumerate(zip(results0, results1)):
                    if num_iters > 1:
                        G_LOGGER.info("Iteration: {:}".format(iteration))
                    with contextlib.ExitStack() as stack:
                        if num_iters > 1:
                            stack.enter_context(G_LOGGER.indent())
                        iteration_match_dict = compare_func(result0, result1)
                        accuracy_result[runner_pair].append(iteration_match_dict)

                    mismatched_outputs = find_mismatched(iteration_match_dict)
                    if fail_fast and mismatched_outputs:
                        return accuracy_result

                passed, failed, total = accuracy_result.stats(runner_pair)
                pass_rate = accuracy_result.percentage(runner_pair) * 100.0
                G_LOGGER.verbose("Finished comparing {:} with {:}".format(runner0_name, runner1_name,))
                if num_iters > 1 or len(comparisons) > 1:
                    msg = "Accuracy Summary | Passed: {:}/{:} iterations | Pass Rate: {:}%".format(
                            passed, total, pass_rate)
                    if passed == total:
                        G_LOGGER.success(msg)
                    else:
                        G_LOGGER.error(msg)
        return accuracy_result


    @staticmethod
    def validate(run_results, check_finite=None, check_nan=None, fail_fast=None):
        """
        Checks output validity.

        Args:
            run_results (Dict[str, List[IterationResult]]): The result of Comparator.run().
            check_finite (bool): Whether to fail on non-finite values. Defaults to False.
            check_nan (bool): Whether to fail on NaNs. Defaults to True.
            fail_fast (bool): Whether to fail after the first invalid value. Defaults to False.

        Returns:
            bool: True if all outputs were valid, False otherwise.
        """
        check_finite = misc.default_value(check_finite, False)
        check_nan = misc.default_value(check_nan, True)
        fail_fast = misc.default_value(fail_fast, False)


        def is_finite(output):
            non_finite = np.logical_not(np.isfinite(output))
            if np.any(non_finite):
                G_LOGGER.error("Encountered one or more non-finite values")
                G_LOGGER.error("Note: Use -vv or set logging verbosity to EXTRA_VERBOSE to display non-finite values", mode=LogMode.ONCE)
                G_LOGGER.extra_verbose("Note: non-finite values at:\n{:}".format(non_finite))
                G_LOGGER.extra_verbose("Note: non-finite values:\n{:}".format(output[non_finite]))
                return False
            return True


        def is_not_nan(output):
            nans = np.isnan(output)
            if np.any(nans):
                G_LOGGER.error("Encountered one or more NaNs")
                G_LOGGER.error("Note: Use -vv or set logging verbosity to EXTRA_VERBOSE to display locations of NaNs", mode=LogMode.ONCE)
                G_LOGGER.extra_verbose("Note: NaNs at:\n{:}".format(nans))
                return False
            return True


        all_valid = True
        for runner_name, results in run_results.items():
            for result in results:
                for output_name, output in result.items():
                    G_LOGGER.info("Runner: {:40} | Validating output: {:} (check_finite={:}, check_nan={:})".format(
                                        runner_name, output_name, check_finite, check_nan))

                    output_valid = True
                    with G_LOGGER.indent():
                        if check_nan:
                            output_valid &= is_not_nan(output)
                        if check_finite:
                            output_valid &= is_finite(output)

                        all_valid &= output_valid

                        if output_valid:
                            G_LOGGER.success("Runner: {:40} | Output: {:} is valid".format(runner_name, output_name))
                        else:
                            G_LOGGER.error("Runner: {:40} | Errors detected in output: {:}".format(runner_name, output_name))
                            if fail_fast:
                                return False

        if all_valid:
            G_LOGGER.success("Validation passed")
        else:
            G_LOGGER.error("Validation failed")
        return all_valid
