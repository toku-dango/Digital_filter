import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter

class MA_Filter():
    def __init__(self, master=None):
        print("MA_Filter")

    def character_MA(self, N, fs, a=1):
        b = np.ones(N) / N
        fn = fs / 2# ナイキスト周波数計算

        w, h = signal.freqz(b, a)# 周波数応答計算
        x_freq_rspns = w / np.pi * fn
        y_freq_rspns = db(abs(h))#複素数をデシベル変換

        p = np.angle(h) * 180 / np.pi#位相特性
        w, gd = signal.group_delay([b, a])# 群遅延計算
        x_gd = w / np.pi * fn
        y_gd = gd

        return x_freq_rspns, y_freq_rspns, p, x_gd, y_gd

    def MA_filter(self, x, move_num):
        move_ave = []
        ring_buff = np.zeros(move_num)
        for i in range(len(x)):
            num = i % move_num
            if i < move_num:
                ring_buff[num] = x[i]
                move_ave.append(0)
            else:
                ring_buff[num] = x[i]
                ave = np.mean(ring_buff)
                move_ave.append(ave)
        return move_ave

class Butter_Filter():
    def __init__(self, master=None):
        print("Butterworse_Filter")

    def get_filtertype(self, selected_num):
        if selected_num == 0:
            ftype1 = "lowpass"
        elif selected_num == 1:
            ftype1 = "highpass"
        else:
            ftype1 = "bandpass"
        return ftype1

    def get_filterparam(self, dt, fl, fh, order, filtertype):
        dt = float(dt)

        fs = 1 / dt
        fn = fs / 2
        if filtertype == "bandpass":
            Wn = [float(fl)/fn, float(fh)/fn]
            N=order//2
        elif filtertype == "lowpass":
            Wn = float(fh)/fn
            N=order
        else:
            Wn = float(fl)/fn
            N=order
        return Wn, N

    def character_Butter(self, dt, order, fl, fh, ftype):
        fs = 1 / dt
        fn = fs / 2# ナイキスト周波数計算
        Wn, N = self.get_filterparam(dt, fl, fh, order, ftype)
        b, a = signal.butter(N, Wn, ftype)#第四引数Trueならばanarog filter/ FalseならばDigital filter
        w, h = signal.freqz(b, a)#アナログフィルタはfreqs,デジタルフィルタはfreqz

        x_freq_rspns = w / np.pi * fn
        y_freq_rspns = db(abs(h))#複素数をデシベル変換
        
        p = np.angle(h) * 180 / np.pi#位相特性

        w, gd = signal.group_delay([b, a])# 群遅延計算
        x_gd = w / np.pi * fn
        y_gd = gd

        return x_freq_rspns, y_freq_rspns, p, x_gd, y_gd

    def Butter_filter(self, data, dt, order, fl, fh, ftype, worN=8192):
        fs = 1 / dt
        fn = fs / 2# ナイキスト周波数計算
        Wn, N = self.get_filterparam(dt, fl, fh, order, ftype)
        print(Wn, N)
        b, a = signal.butter(N, Wn, ftype)#第四引数Trueならばanarog filter/ FalseならばDigital filter
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, data, zi=zi*data[0])

        return z

class Chebyshev_Filter():
    def __init__(self, master=None):
        print("Chebyshev_Filter")

    def get_filtertype(self, selected_num):
        if selected_num == 0:
            ftype1 = "lowpass"
        elif selected_num == 1:
            ftype1 = "highpass"
        else:
            ftype1 = "bandpass"
        return ftype1

    def get_filterparam(self, dt, fl, fh, order, filtertype):
        dt = float(dt)

        fs = 1 / dt
        fn = fs / 2
        if filtertype == "bandpass":
            Wn = [float(fl)/fn, float(fh)/fn]
            N=order//2
        elif filtertype == "lowpass":
            Wn = float(fh)/fn
            N=order
        else:
            Wn = float(fl)/fn
            N=order
        return Wn, N

    def character_Chebyshev(self, dt, order, fl, fh, ripple, ftype):
        fs = 1 / dt
        fn = fs / 2# ナイキスト周波数計算
        Wn, N = self.get_filterparam(dt, fl, fh, order, ftype)
        b, a = signal.cheby1(N, ripple, Wn, ftype)
        w, h = signal.freqz(b, a)#アナログフィルタはfreqs,デジタルフィルタはfreqz

        x_freq_rspns = w / np.pi * fn
        y_freq_rspns = db(abs(h))#複素数をデシベル変換
        
        p = np.angle(h) * 180 / np.pi#位相特性

        w, gd = signal.group_delay([b, a])# 群遅延計算
        x_gd = w / np.pi * fn
        y_gd = gd

        return x_freq_rspns, y_freq_rspns, p, x_gd, y_gd

    def Chebyshev_filter(self, data, dt, order, fl, fh, ripple, ftype):
        fs = 1 / dt
        fn = fs / 2# ナイキスト周波数計算
        Wn, N = self.get_filterparam(dt, fl, fh, order, ftype)
        b, a = signal.cheby1(N, ripple, Wn, ftype)#第四引数Trueならばanarog filter/ FalseならばDigital filter

        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, data, zi=zi*data[0])

        return z


class Chebyshev_second_Filter():
    def __init__(self, master=None):
        print("Chebyshev_Filter")

    def get_filtertype(self, selected_num):
        if selected_num == 0:
            ftype1 = "lowpass"
        elif selected_num == 1:
            ftype1 = "highpass"
        else:
            ftype1 = "bandpass"
        return ftype1

    def get_filterparam(self, dt, fl, fh, order, filtertype):
        dt = float(dt)

        fs = 1 / dt
        fn = fs / 2
        if filtertype == "bandpass":
            Wn = [float(fl)/fn, float(fh)/fn]
            N=order//2
        elif filtertype == "lowpass":
            Wn = float(fh)/fn
            N=order
        else:
            Wn = float(fl)/fn
            N=order
        return Wn, N

    def character_Chebyshev_second(self, dt, order, fl, fh, attenuation, ftype):
        fs = 1 / dt
        fn = fs / 2# ナイキスト周波数計算
        Wn, N = self.get_filterparam(dt, fl, fh, order, ftype)
        b, a = signal.cheby2(N, attenuation, Wn, ftype)
        w, h = signal.freqz(b, a)#アナログフィルタはfreqs,デジタルフィルタはfreqz

        x_freq_rspns = w / np.pi * fn
        y_freq_rspns = db(abs(h))#複素数をデシベル変換
        
        p = np.angle(h) * 180 / np.pi#位相特性

        w, gd = signal.group_delay([b, a])# 群遅延計算
        x_gd = w / np.pi * fn
        y_gd = gd

        return x_freq_rspns, y_freq_rspns, p, x_gd, y_gd

    def Chebyshev_second_filter(self, data, dt, order, fl, fh, attenuation, ftype):
        fs = 1 / dt
        fn = fs / 2# ナイキスト周波数計算
        Wn, N = self.get_filterparam(dt, fl, fh, order, ftype)
        b, a = signal.cheby2(N, attenuation, Wn, ftype)

        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, data, zi=zi*data[0])

        return z

class Bessel_Filter():
    def __init__(self, master=None):
        print("Chebyshev_Filter")

    def get_filtertype(self, selected_num):
        if selected_num == 0:
            ftype1 = "lowpass"
        elif selected_num == 1:
            ftype1 = "highpass"
        else:
            ftype1 = "bandpass"
        return ftype1

    def get_filterparam(self, dt, fl, fh, order, filtertype):
        dt = float(dt)

        fs = 1 / dt
        fn = fs / 2
        if filtertype == "bandpass":
            Wn = [float(fl)/fn, float(fh)/fn]
            N=order//2
        elif filtertype == "lowpass":
            Wn = float(fh)/fn
            N=order
        else:
            Wn = float(fl)/fn
            N=order
        return Wn, N

    def character_Bessel(self, dt, order, fl, fh, ftype):
        fs = 1 / dt
        fn = fs / 2# ナイキスト周波数計算
        Wn, N = self.get_filterparam(dt, fl, fh, order, ftype)
        b, a = signal.bessel(N, Wn, ftype)
        w, h = signal.freqz(b, a)#アナログフィルタはfreqs,デジタルフィルタはfreqz

        x_freq_rspns = w / np.pi * fn
        y_freq_rspns = db(abs(h))#複素数をデシベル変換

        p = np.angle(h) * 180 / np.pi#位相特性

        w, gd = signal.group_delay([b, a])# 群遅延計算
        x_gd = w / np.pi * fn
        y_gd = gd

        return x_freq_rspns, y_freq_rspns, p, x_gd, y_gd

    def Bessel_filter(self, data, dt, order, fl, fh, ftype, worN=8192):
        fs = 1 / dt
        fn = fs / 2# ナイキスト周波数計算
        Wn, N = self.get_filterparam(dt, fl, fh, order, ftype)
        b, a = signal.bessel(N, Wn, ftype)#第四引数Trueならばanarog filter/ FalseならばDigital filter

        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, data, zi=zi*data[0])

        return z

def FFT(Raw_Data, sampling_time):
    #sampling_time = 0.005
    sampling_time_f = 1/sampling_time
    fft_nseg = 256
    fft_overlap = fft_nseg // 2
    f_raw, t_raw, Sxx_raw = signal.stft(Raw_Data, sampling_time_f, nperseg=fft_nseg, noverlap=fft_overlap)

    return f_raw, t_raw, Sxx_raw

def get_start_end_index(t, start_t, end_t):
    start_ind=0
    while(t[start_ind] < start_t):
        start_ind += 1
    end_ind = start_ind + 1
    while (t[end_ind] < end_t):
        end_ind += 1
    return start_ind, end_ind

def ave_spectrum(Sxx, t, start_t, end_t):
    start_ind, end_ind = get_start_end_index(t, start_t, end_t)
    ave_spectrum = np.zeros(Sxx.shape[0])
    for i in range(Sxx.shape[0]):
        ave_spectrum[i] = np.average(Sxx[i, start_ind:end_ind])
    return ave_spectrum

def db(x, dBref=1):  # デシベル変換
    y = 20 * np.log10(x / dBref)
    return y
