from multiprocessing import Process, pool, Pool

__author__ = 'jumingxing'


class NoDaemonProcess (Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property (_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.

class MyPool (pool.Pool):
    Process = NoDaemonProcess


def myPool(n_jobs, arg, function):
    pool_1 = MyPool (processes=n_jobs)
    result = pool_1.map (function, arg)
    pool_1.close ()
    pool_1.join ()
    return result


def originalPool(n_jobs, arg, function):
    pool_1 = Pool (processes=n_jobs)
    result = pool_1.map (function, arg)
    pool_1.close ()
    pool_1.join ()
    return result