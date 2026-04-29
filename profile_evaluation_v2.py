from pylib_general import gaussian_convolution_nonuniform
import numpy as np
import pandas as pd
from optimize_v2 import run_optimization
from PyQt5 import QtCore
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import get_context
from concurrent.futures import wait, FIRST_COMPLETED
from datetime import datetime
import os
import traceback
import matplotlib
from matplotlib.pyplot import close
from scipy.stats import linregress
import csv

matplotlib.use("Agg")

def process_h5_data(job, config, fig_dir, history=False, progress_cb=None):
    coords = job['coords']
    file_num = job['file_num']
    file_name = job['file_key']
    ds1 = job['df']
    ds1 = ds1[:,np.any(~np.isnan(ds1), axis=0)]

    y1 = np.nanmean(ds1, axis=0)
    n_points = len(y1)
    # start, end = round(0.15*len(y1)), round(0.8*len(y1))
    # y1 = y1[start:end]

    #FIX!!!!!!!!
    sigma_x = {}
    x = np.arange(n_points)

    if n_points < 450:
        x_interp = np.linspace(0, n_points, 1000)
        y_interp = np.interp(x_interp, x, y1)
        data = np.array([x_interp, y_interp]).T

        x,y1 = x_interp, y_interp
    else:
        data = np.array([x, y1]).T
        
    smooth_y1 = gaussian_convolution_nonuniform(x, y1, sigma_x=2)
    smooth_data = np.array([x, smooth_y1]).T

    table, fig, model, u, losses, init_p = run_optimization(smooth_data, 
                                                            data, 
                                                            file_num, 
                                                            config.sigma, 
                                                            config.rand, 
                                                            config.uncert, 
                                                            coords, 
                                                            config.w_bounds)
    #self.init_p = init_p if self.prev_ip else None


    if config.save_loc:
        fig_path = os.path.join(fig_dir, f"{file_name}_profile{file_num}.png")
        fig.savefig(fig_path)
        close(fig)

    return {'file_key': job['file_key'], 'file_num': file_num,
                'res': {'table': table, 'fig': fig, 'losses': losses, 
                        'init_p': init_p, 'uncert': u, 'file_info': job['file_info']}}


def process_txt_data(job, config, fig_dir, history=False, progress_cb=None):
        df = job['df']
        file_num = job['file_num']
        file_name = job['file_key']
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
        if config.save_loc:
            fig_path = os.path.join(fig_dir, f"{file_name}_profile{file_num}.png")
            fig.savefig(fig_path)
            close(fig)

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
            fig_dir = None
            csv_losses = "test_folder/losses.csv"


            if self.config.save_loc:
                csv_path = os.path.join(self.config.save_loc, "results.csv")
                write_header = True
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                fig_dir = os.path.join(self.config.save_loc, f"figures_{timestamp}")
                os.makedirs(fig_dir, exist_ok=True)

            #max_workers = max(1, (os.cpu_count() or 1) - 1)
            max_workers = 8
            ctx = get_context("spawn")
            comp_func = process_h5_data if self.config.file_format == '.h5' else process_txt_data

            with open(csv_losses, "a", newline="") as f:
                writer = csv.writer(f)

                with ProcessPoolExecutor(
                    mp_context=ctx,
                    max_workers=max_workers
                ) as pool:

                    futures = set()
                    job_iter = iter(self.jobs)

                    for _ in range(max_workers):
                        try:
                            job = next(job_iter)
                            futures.add(
                                pool.submit(comp_func, job, self.config, fig_dir)
                            )
                        except StopIteration:
                            break

                    while futures:

                        self.lock_signal.lock()
                        abort = self.abort_signal
                        paused = self.pause_signal
                        self.lock_signal.unlock()

                        if abort:
                            pool.shutdown(wait=False, cancel_futures=True)
                            break

                        if paused:
                            self.paused.emit()
                            self.lock_signal.lock()
                            while self.pause_signal and not self.abort_signal:
                                self.wait_signal.wait(self.lock_signal)
                            self.lock_signal.unlock()
                            self.resumed.emit()
                            continue

                        done, futures = wait(
                            futures,
                            return_when=FIRST_COMPLETED
                        )

                        for fut in done:

                            if self.abort_signal:
                                pool.shutdown(wait=False, cancel_futures=True)
                                break

                            try:
                                result = fut.result()
                                results.append(result)

                                writer.writerow([result["file_num"],
                                                 result["res"]["losses"]["total_loss"][-1]])

                                if self.config.save_loc:
                                    curr_table = result["res"]["table"]
                                    curr_table.to_csv(csv_path, mode='a', header=write_header, index=False)
                                    write_header = False

                                completed += 1
                                self.progress.emit(completed)

                            except Exception:
                                self.error.emit(traceback.format_exc())

                            try:
                                job = next(job_iter)
                                futures.add(
                                    pool.submit(comp_func, job, self.config, fig_dir)
                                )
                            except StopIteration:
                                pass

            if not self.abort_signal:
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


