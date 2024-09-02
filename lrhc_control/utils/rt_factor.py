import time

class RtFactor():

    def __init__(self,
            dt_nom: float,
            window_size: int):

        self._it_counter = 0
        
        self._dt_nom = dt_nom

        self._start_time = time.perf_counter()

        self._current_rt_factor = 0.0

        self._window_size = window_size

        self._real_time = 0
        self._nom_time = 0

    def update(self):

        self._real_time = time.perf_counter() - self._start_time

        self._it_counter += 1

        self._nom_time += self._dt_nom

        self._current_rt_factor = self._nom_time / self._real_time

    def reset_due(self):

        return (self._it_counter+1) % self._window_size == 0
    
    def get_avrg_step_time(self):

        return self._real_time / self._window_size
    
    def get_dt_nom(self):

        return self._dt_nom
    
    def get_nom_time(self):

        return self._now_time

    def get(self):

        return self._current_rt_factor
    
    def reset(self):

        self._it_counter = 0
        self._nom_time = 0
        self._start_time = time.perf_counter()