from pylib_general import gaussian_convolution_nonuniform
import numpy as np
from optimize_v2 import run_optimization
from PyQt5 import QtCore
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import get_context
import os
import traceback
import matplotlib

matplotlib.use("Agg")


def eval_data_comp(job, config, history=False, progress_cb=None):
        df = job['df']
        file_num = job['file_num']
        coords = job['coords']

        # FIX SIGMA_X!!!
        y1 = gaussian_convolution_nonuniform(df[config.strike], df[config.parallel], sigma_x=5)
        smooth_data = np.vstack([df[config.strike], y1]).T

        if config.n_dim > 1:
            y2 = gaussian_convolution_nonuniform(df[config.strike], df[config.perp], sigma_x=5)
            smooth_data = np.hstack([smooth_data, y2[np.newaxis, :].T])

        df_data = df[[config.strike, config.parallel]] if config.n_dim < 2 else df[[config.strike, config.parallel, config.perp]]
        
        if history:
            return run_optimization(smooth_data, df_data.to_numpy(), file_num, config.sigma, config.rand, history=history, progress_cb=progress_cb)

        table, fig, model, u, losses, init_p = run_optimization(smooth_data, 
                                                                df_data.to_numpy(), 
                                                                file_num, 
                                                                config.sigma, 
                                                                config.rand, 
                                                                config.uncert, 
                                                                coords, 
                                                                config.w_bounds,
                                                                progress_cb)
        #config.init_p = init_p if config.prev_ip else None

        return {'file_key': job['file_key'], 'file_num': file_num,
                'res': {'table': table, 'fig': fig, 'losses': losses, 
                        'init_p': init_p, 'uncert': u, 'file_info': job['file_info']}}


class EvalCoordinator(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)
    paused = QtCore.pyqtSignal()
    resumed = QtCore.pyqtSignal()

    def __init__(self, jobs, config):
        super().__init__()
        self.jobs = jobs
        self.config = config

        self.pause_signal = False
        self.abort_signal = False
        self.lock_signal = QtCore.QMutex()
        self.wait_signal = QtCore.QWaitCondition()

    @QtCore.pyqtSlot()
    def run(self):
        print(">>> EvalCoordinator.run entered")

        results = []
        completed = 0

        try:
            ctx = get_context("spawn")

            #with ThreadPoolExecutor(
            with ProcessPoolExecutor(
                mp_context=ctx,
                max_workers=os.cpu_count() - 1
            ) as pool:

                futures = []
                job_iter = iter(self.jobs)

                while True:
                    self.lock_signal.lock()
                    if self.abort_signal:
                        self.lock_signal.unlock()
                        break

                    if self.pause_signal:
                        self.paused.emit()
                        self.wait_signal.wait(self.lock_signal)
                        self.resumed.emit()

                    self.lock_signal.unlock()

                    try:
                        job = next(job_iter)
                    except StopIteration:
                        break

                    futures.append(
                        pool.submit(eval_data_comp, job, self.config)
                    )

                for fut in as_completed(futures):
                    if self.abort_signal:
                        break

                    results.append(fut.result())

                    completed += 1
                    self.progress.emit(completed)

            self.finished.emit(results)

        except Exception:
            self.error.emit(traceback.format_exc())

    @QtCore.pyqtSlot()
    def pause(self):
        self.lock_signal.lock()
        self.pause_signal = True
        self.lock_signal.unlock()

    @QtCore.pyqtSlot()
    def resume(self):
        self.lock_signal.lock()
        self.pause_signal = False
        self.wait_signal.wakeAll()
        self.lock_signal.unlock()

    @QtCore.pyqtSlot()
    def cancel(self):
        self.lock_signal.lock()
        self.abort_signal = True
        self.wait_signal.wakeAll()
        self.lock_signal.unlock()


