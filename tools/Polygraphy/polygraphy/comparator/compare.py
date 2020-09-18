from collections import OrderedDict

import numpy as np
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.util import misc


class OutputCompareResult(object):
    """
    Represents the result of comparing a single output of a single iteration
    between two runners.
    """
    def __init__(self, passed, required_atol, required_rtol):
        """
        Records the required tolerances for the results to be considered equivalent.

        Args:
            passed (bool): Whether the error was within acceptable limits.
            required_atol (float): The minimum required absolute tolerance to consider the outputs equivalent.
            required_rtol (float): The minimum required relative tolerance to consider the outputs equivalent.
        """
        self.passed = passed
        self.required_atol = required_atol
        self.required_rtol = required_rtol


    def __bool__(self):
        """
        Whether the output matched.

        Returns:
            bool
        """
        return self.passed


    def __str__(self):
        return "(atol={:}, rtol={:})".format(self.required_atol, self.required_rtol)


# Provides functions to compare two IterationResults
class CompareFunc(object):
    """
    Provides functions that can be used to compare two `IterationResult` s.
    """

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
            Callable(IterationResult, IterationResult) -> OrderedDict[str, OutputCompareResult]:
                A callable that returns a mapping of output names to `OutputCompareResult` s, indicating
                whether the corresponding output matched.
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
                G_LOGGER.error("FAILED | Mismatched outputs: {:}".format(mismatched_output_names))

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
