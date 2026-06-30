from pylib_general import gaussian_convolution_nonuniform
import numpy as np
from optimize_v3 import run_optimization
from PyQt5 import QtCore
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import get_context
from concurrent.futures import wait, FIRST_COMPLETED, CancelledError
from datetime import datetime
import os
import traceback
import matplotlib
from matplotlib.pyplot import close
import csv

matplotlib.use("Agg")

def process_h5_data(job, config, fig_dir, history=False, progress_cb=None):
    coords = job['coords']
    file_num = job['file_num']
    file_name, _, _ = job['file_key']
    ds1 = job['df1']
    ds2 = job['df2']
    ys = []

    ds1 = ds1[:,np.any(~np.isnan(ds1), axis=0)]
    ys.append(np.nanmean(ds1, axis=0))
    if ds2 is not None:
        ds2 = ds2[:,np.any(~np.isnan(ds2), axis=0)]
        ys.append(np.nanmean(ds2, axis=0))
   
    n_points = len(ys[0])

    #FIX!!!!!!!!
    sigma_x = {}
    x = np.arange(n_points)

    if n_points < 450:
        x_interp = np.linspace(0, n_points-1, 1000)
        ys = [np.interp(x_interp, x, y) for y in ys]
        x = x_interp
    
    smooth_ys = [gaussian_convolution_nonuniform(x, y, sigma_x=20)
                 for y in ys]
    
    data = np.column_stack((x, *ys))
    smooth_data = np.column_stack((x, *smooth_ys))

    table, fig, model, u, losses, init_p = run_optimization(smooth_data, 
                                                            data, 
                                                            file_num, 
                                                            config.sigma, 
                                                            config.rand, 
                                                            config.uncert, 
                                                            coords, 
                                                            config.w_bounds)
    #self.init_p = init_p if self.prev_ip else None

    for n, p in model.named_parameters():
        print(n, p)

    if config.save_loc:
        save_name = file_name+".png" if file_name == file_num else f"{file_name}_profile{file_num}.png"
        fig_path = os.path.join(fig_dir, save_name)
        fig.savefig(fig_path)
        close(fig)

    return {'file_key': job['file_key'], 'file_num': file_num, 'table': table, 
            'fig': fig, 'losses': losses, 'init_p': init_p, 'uncert': u, 
            'file_info': job['file_info']}


def process_txt_data(job, config, fig_dir, history=False, progress_cb=None):
        df = job['df']
        file_num = job['file_num']
        file_name, _, _ = job['file_key']
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

        return {'file_key': job['file_key'], 'file_num': file_num, 'table': table,
                'fig': fig, 'losses': losses, 'init_p': init_p, 'uncert': u, 
                'file_info': job['file_info']}


class EvalCoordinator(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    result_ready = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)
    paused = QtCore.pyqtSignal()
    resumed = QtCore.pyqtSignal()

    def __init__(self, jobs, config):
        super().__init__()
        self.jobs = jobs
        self.config = config

        self.pool = None
        self.futures = set()

        self.pause_signal = False
        self.abort_signal = False
        self.shutdown_signal = False
        self.lock_signal = QtCore.QMutex()
        self.wait_signal = QtCore.QWaitCondition()

    def _read_flags(self):
        self.lock_signal.lock()
        abort = self.abort_signal
        paused = self.pause_signal
        shutting_down = self.shutdown_signal
        self.lock_signal.unlock()
        return abort, paused, shutting_down

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
                    
                    self.pool = pool
                    self.futures = set()

                    job_iter = iter(self.jobs)

                    for _ in range(max_workers):
                        try:
                            job = next(job_iter)
                            self.futures.add(
                                pool.submit(comp_func, job, self.config, fig_dir)
                            )
                        except StopIteration:
                            break

                    while self.futures:

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

                        done, remaining = wait(
                            self.futures,
                            return_when=FIRST_COMPLETED
                        )

                        self.futures = remaining

                        for fut in done:

                            abort, paused, shutting_down = self._read_flags()

                            if abort:
                                break

                            try:
                                result = fut.result()
                            except CancelledError:
                                # expected during cancel or shutdown
                                continue
                            
                            except Exception:
                                abort, paused, shutting_down = self._read_flags()

                                if not abort and not shutting_down:
                                    self.error.emit(traceback.format_exc())
                                continue

                            abort, paused, shutting_down = self._read_flags()

                            if not abort and not shutting_down:
                                results.append(result)

                                writer.writerow([result["file_num"],
                                                result["losses"]["total_loss"][-1]])

                                if self.config.save_loc:
                                    curr_table = result["table"]
                                    curr_table.to_csv(csv_path, mode='a', header=write_header, index=False)
                                    write_header = False

                                completed += 1
                                
                                self.result_ready.emit(result)
                                self.progress.emit(completed)

                            # Submit another job only if:
                            # - not aborting
                            # - not shutting down
                            # - not paused
                            abort, paused, shutting_down = self._read_flags()

                            if not abort and not paused and not shutting_down:
                                try:
                                    job = next(job_iter)

                                    self.futures.add(
                                        pool.submit(comp_func, job, self.config, fig_dir)
                                    )
                                except StopIteration:
                                    pass
                        
                        if self.abort_signal:
                            break
            self.pool = None
            self.futures = set()
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
        self.shutdown_signal = False
        self.pause_signal = False
        self.wait_signal.wakeAll()
        self.lock_signal.unlock()

        # Cancel queued futures
        try:
            for fut in list(getattr(self, "futures", [])):
                fut.cancel()
        except Exception:
            pass

        # Stop the process pool
        try:
            if getattr(self, "pool", None) is not None:
                self.pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


    @QtCore.pyqtSlot()
    def shutdown_after_running_jobs(self):
        """
        GUI is closing.

        Behavior:
        - Do not submit any more jobs.
        - Do not emit completed profile results to the GUI.
        - Let already-running run_optimization calls finish.
        - Then emit finished so the GUI can close.
        """
        self.lock_signal.lock()
        self.shutdown_signal = True
        self.pause_signal = False
        self.wait_signal.wakeAll()
        self.lock_signal.unlock()

        # Cancel futures that have not started yet.
        # Running futures will continue until run_optimization returns.
        try:
            for fut in list(getattr(self, "futures", [])):
                fut.cancel()
        except Exception:
            pass