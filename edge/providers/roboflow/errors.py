from edge.harness.errors import CapabilityUnavailableError, RetryableExecutionError, HarnessError


class ProviderError(HarnessError):
    pass


class ProviderCapabilityUnavailable(CapabilityUnavailableError):
    pass


class ProviderRetryableError(RetryableExecutionError):
    pass
