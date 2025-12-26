# complex_langevin/utils/logging.py
class SimLogger:
    def log(self, message):
        self._sender = None
        pass

    def _log_enabled(self, message):
        print(f"[{self._sender}] {message}")

    def _log_disabled(self, message):
        pass

    def _init_logger(self, verbose: bool, sender: str = "Logger"):
        self._sender = sender
        self.log =  self._log_enabled if verbose else self._log_disabled