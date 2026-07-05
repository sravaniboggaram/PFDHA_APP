from pylib_general import gaussian_convolution_nonuniform
import numpy as np
from optimize_v3 import run_optimization
from PyQt5 import QtCore
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from concurrent.futures import wait, FIRST_COMPLETED, CancelledError
from datetime import datetime
import os
import traceback
from matplotlib.pyplot import close, subplots
import pandas as pd
from h5py import File as h5pyFile


def save_figures(fig, losses, config_save_loc, save_loc_fig_dir, config_temp_fig_fold, file_name, file_num):
    save_name = file_name+".png" if file_name == file_num else f"{file_name}_profile{file_num}.png"
    if config_save_loc is not None:
        fig_path = os.path.join(save_loc_fig_dir, save_name)
    else:
        fig_path = os.path.join(config_temp_fig_fold, save_name)

    fig.savefig(fig_path)
    close(fig)

    fig_loss_path = os.path.join(config_temp_fig_fold, "LOSS_" + save_name)
    fig_loss, ax = subplots(1, 1)
    ax.plot(range(1, len(losses['total_loss']) + 1), losses['total_loss'])
    ax.set_title("Loss vs Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss -> L1 Loss -> MSE Loss ")
    fig_loss.savefig(fig_loss_path)
    close(fig)

    return fig_path, fig_loss_path


def process_smoothed_data(x, ys, config, file_name, file_num, coords, fig_dir, device):
    n_points = len(x)
    orig_data = np.column_stack((x, *ys))

    if n_points < 450 or config.interp is not None:
        n_interp_points = config.interp if config.interp is not None else 1000
        x_interp = np.linspace(0, n_points-1, n_interp_points)
        ys = [np.interp(x_interp, x, y) for y in ys]
        x = x_interp
    
    smooth_ys = [gaussian_convolution_nonuniform(x, y, sigma_x=20)
                 for y in ys]
    
    smooth_data = np.column_stack((x, *smooth_ys))

    table, fig, model, u, losses, init_p = run_optimization(smooth_data, 
                                                            orig_data, 
                                                            file_num, 
                                                            config.sigma, 
                                                            config.rand, 
                                                            config.uncert, 
                                                            coords, 
                                                            config.w_bounds,
                                                            device=device)

    fig_path, fig_loss_path = save_figures(fig, losses, config.save_loc,
                                           fig_dir, config.temp_fig_folder,
                                           file_name, file_num)


    return fig_path, fig_loss_path, table, u, init_p, losses['total_loss'][-1]


def process_h5_data(job, config, fig_dir, device, history=False):
    coords = job['coords']
    file_num = job['file_num']
    file_name, _, _ = job['file_key']
    df_path = job['df_path']
    prof_key = job['prof_key']
    ys = []

    f = h5pyFile(df_path, 'r')
    groups = [f[k] for k in list(f.keys())]

    ds1 = np.array(groups[0][prof_key])
    ds2 = np.array(groups[1][prof_key]) if len(groups) == 2 else None

    ds1 = ds1[:,np.any(~np.isnan(ds1), axis=0)]
    ys.append(np.nanmean(ds1, axis=0))
    if ds2 is not None:
        ds2 = ds2[:,np.any(~np.isnan(ds2), axis=0)]
        ys.append(np.nanmean(ds2, axis=0))
   
    x = np.arange(len(ys[0]))

    fig_path, fig_loss_path, table, u, init_p, final_loss = process_smoothed_data(x,
                                                                                  ys,
                                                                                  config,
                                                                                  file_name,
                                                                                  file_num,
                                                                                  coords,
                                                                                  fig_dir,
                                                                                  device)

    return {'file_key': job['file_key'], 'file_num': file_num, 'table': table, 
            'fig': fig_path, 'losses': fig_loss_path, 'init_p': init_p, 'uncert': u, 
            'file_info': job['file_info'], 'final_loss': final_loss}


def process_txt_data(job, config, fig_dir, device, history=False):
        df_path = job['df']
        file_num = job['file_num']
        file_name, _, file_idx = job['file_key']
        coords = job['coords']
        cols, new_names, _, delim, header = config.txt_cols_data

        data = pd.read_csv(df_path, delimiter=delim, header=header)
        df = data[cols].rename(columns=new_names)

        if file_idx is not None:
            profiles = df.groupby(config.ids)
            df = profiles.get_group(file_num)

        x = df[config.strike].to_numpy()
        ys = [df[config.parallel].to_numpy()]
        if config.n_dim > 1:
            ys.append(df[config.perp].to_numpy())

        
        fig_path, fig_loss_path, table, u, init_p, final_loss = process_smoothed_data(x,
                                                                                      ys,
                                                                                      config,
                                                                                      file_name,
                                                                                      file_num,
                                                                                      coords,
                                                                                      fig_dir,
                                                                                      device)
        

        return {'file_key': job['file_key'], 'file_num': file_num, 'table': table,
                'fig': fig_path, 'losses': fig_loss_path, 'init_p': init_p, 'uncert': u, 
                'file_info': job['file_info'], 'final_loss': final_loss}

class EvalCoordinator(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    result_ready = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)
    paused = QtCore.pyqtSignal()
    resumed = QtCore.pyqtSignal()

    def __init__(self, jobs, config, device, n_cores):
        super().__init__()
        self.jobs = jobs
        self.config = config
        self.device = device
        self.n_cores = n_cores
        self.curr_gpu_id = 0

        print("self device ", device)

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

            if self.config.save_loc:
                csv_path = os.path.join(self.config.save_loc, "results.csv")
                write_header = True
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                fig_dir = os.path.join(self.config.save_loc, f"figures_{timestamp}")
                #os.makedirs(fig_dir, exist_ok=True)
                os.makedirs(fig_dir)

            #max_workers = 8
            max_workers = self.n_cores
            ctx = get_context("spawn")
            comp_func = process_h5_data if self.config.file_format == '.h5' else process_txt_data

            with ProcessPoolExecutor(
                mp_context=ctx,
                max_workers=max_workers
            ) as pool:
                
                self.pool = pool
                self.futures = set()

                job_iter = iter(self.jobs)

                for _ in range(max_workers):
                    try:
                        if self.device == "cuda":
                            device_id = f"cuda:{self.curr_gpu_id}"
                            if self.n_cores != 1:
                                self.curr_gpu_id = (self.curr_gpu_id + 1) % self.n_cores
                        else:
                            device_id = "cpu"

                        job = next(job_iter)
                        self.futures.add(
                            pool.submit(comp_func, job, self.config, fig_dir, device_id)
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

                            # writer.writerow([result["file_num"],
                            #                 result["losses"]["total_loss"][-1]])

                            if self.config.save_loc:
                                curr_table = result["table"]
                                curr_table.to_csv(csv_path, mode='a', header=write_header, index=False)
                                write_header = False

                            completed += 1
                            
                            self.result_ready.emit(result)
                            self.progress.emit(completed)

                        abort, paused, shutting_down = self._read_flags()

                        if not abort and not paused and not shutting_down:
                            try:
                                job = next(job_iter)

                                if self.device == "cuda":
                                    device_id = f"cuda:{self.curr_gpu_id}"
                                    if self.n_cores != 1:
                                        self.curr_gpu_id = (self.curr_gpu_id + 1) % self.n_cores
                                else:
                                    device_id = "cpu"

                                self.futures.add(
                                    pool.submit(comp_func, job, self.config, fig_dir, device_id)
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