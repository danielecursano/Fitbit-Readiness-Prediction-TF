import pandas as pd
import functools as ft
import os

class Vectorizer:
    """
    Return np.array of shape (11, ) [
        sum(azm), 
        hrv.rmssd, 
        hrv.nremh, 
        hrv.entropy, 
        spo2.average value, 
        sleep.overall_score,
        sleep.revitalization_score,
        sleep.deep_sleep_in_minutes,
        sleep.resting_heart_rate,
        sleep.restlessness,
        stress_score]
    """
    def __init__(self, path):
        self.path = path

    def vectorize_azm(self):
        def mask(file):
            df = pd.read_csv(file)
            df['date'] = pd.to_datetime(df['date_time'])
            df.drop(columns=['date_time'], inplace=True)
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            return df.groupby("date")["total_minutes"].sum().to_frame()
        dir = f"{self.path}/Active Zone Minutes (AZM)/"
        azm_data = None
        files = os.listdir(dir)
        for file in files:
            tmp = mask(f"{dir}{file}")
            if azm_data is None:
                azm_data = tmp
            else:
                azm_data = pd.concat([azm_data, tmp])
        azm_data.reset_index(inplace=True)
        return azm_data
    
    def vectorize_hrv(self):
        dir = f"{self.path}/Heart Rate Variability/"
        files = os.listdir(dir)
        hrv_data = None
        for file in files:
            if file.startswith("Daily Heart Rate Variability Summary") and file[len(file)-3:] == "csv":
                df = pd.read_csv(f"{dir}{file}")
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime('%Y-%m-%d')
                if hrv_data is None:
                    hrv_data = df
                else:
                    hrv_data = pd.concat([hrv_data, df])
        return hrv_data
    
    def vectorize_spo2(self):
        dir = f"{self.path}/Oxygen Saturation (SpO2)/"
        files = os.listdir(dir)
        os_data = None
        for file in files:
            if file.startswith("Daily SpO2 - "):
                df = pd.read_csv(f"{dir}{file}")
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
                df.drop(["lower_bound", "upper_bound"], axis=1, inplace=True)
                if os_data is None:
                    os_data = df
                else:
                    os_data = pd.concat([os_data, df])
        return os_data
    
    def vectorize_sleep(self):
        dir = f"{self.path}/Sleep Score/sleep_score.csv"
        df = pd.read_csv(dir)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
        df.drop(["composition_score", "duration_score", "sleep_log_entry_id"], axis=1, inplace=True)
        return df
    
    def vectorize_stress(self):
        dir = f"{self.path}/Stress Score/Stress Score.csv"
        df = pd.read_csv(dir)
        df = df[~df['CALCULATION_FAILED']]
        df = df[["DATE", "STRESS_SCORE"]]
        df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d')
        return df

    def vectorize(self):
        tmp_data = [self.vectorize_azm(), self.vectorize_hrv(), self.vectorize_spo2(), self.vectorize_sleep(), self.vectorize_stress()]
        new_names = {"timestamp": "date", "DATE": "date"}
        for i in tmp_data:
            i.rename(columns=new_names, inplace=True)
        df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='date'), tmp_data)  
        return df_final.values[:,1:]