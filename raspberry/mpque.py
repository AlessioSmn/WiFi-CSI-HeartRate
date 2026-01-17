from collections import deque
from multiprocessing import Condition, Lock
import time

# MultiProcessDeque
class MPque:
    def __init__(self, maxlen=None):
        self._deque = deque(maxlen=maxlen)
        self._lock = Lock()
        self._cond = Condition(self._lock)

    def appendright(self, item):
        with self._cond:
            self._deque.append(item)
            self._cond.notify()

    def appendleft(self, item):
        with self._cond:
            self._deque.appendleft(item)
            self._cond.notify()

    def pop(self, block=True, timeout=None):
        return self._pop_generic(left=False, block=block, timeout=timeout)

    def popleft(self, block=True, timeout=None):
        return self._pop_generic(left=True, block=block, timeout=timeout)

    def _pop_generic(self, left, block, timeout):
        with self._cond:
            if not block:
                if not self._deque:
                    raise IndexError("pop from empty deque")
            else:
                end = None if timeout is None else time.monotonic() + timeout
                while not self._deque:
                    remaining = None if end is None else end - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("pop timeout")
                    self._cond.wait(remaining)

            return self._deque.popleft() if left else self._deque.pop()

    def __len__(self):
        with self._lock:
            return len(self._deque)
