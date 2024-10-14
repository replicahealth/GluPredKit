from glupredkit.models.base_model import BaseModel
from py_replay_bg.py_replay_bg import ReplayBG
from glupredkit.helpers.scikit_learn import process_data
from scipy.stats import norm
import datetime
import copy
import os
import numpy as np
import pandas as pd
import shutil


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)

        self.models = []
        self.subject_ids = []
        self.bw = None
        self.scenario = None
        self.safe_file_string = None

    def _fit_model(self, x_train, y_train, n_steps=10000, training_samples_per_subject=8000, scenario='multi-meal', *args):
        # n_steps is the number of steps that will be used for identification
        # (for multi-meal it should be at least 100k, for single-meal 10k is default)
        # Note that this class will not work if the dataset does not have five-minute intervals between measurements
        required_columns = ['CGM', 'carbs', 'basal', 'bolus']
        missing_columns = [col for col in required_columns if col not in x_train.columns]
        if missing_columns:
            raise ValueError(
                f"The input DataFrame is missing the following required columns: {', '.join(missing_columns)}")

        x_train = self.process_input_data(x_train)

        relevant_columns = ['id', 'bolus', 'basal', 'glucose', 'cho', 'cho_label', 'bolus_label', 'exercise', 't', 'CR', 'CF']
        x_train = x_train[relevant_columns]

        now = datetime.datetime.now()
        self.safe_file_string = now.strftime("%d%m%Y_%H%M")
        print(self.safe_file_string)

        cwd = os.getcwd()
        print("Creating results directory...")
        # TODO: make the results be stored in the data folder, and in a hidden folder like DNNs
        path = os.path.join(cwd, self.safe_file_string)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory {path}.")

        # Fit parameters of ReplayBG object
        modality = 'identification'
        self.bw = 80  # Placeholder body weight
        self.scenario = scenario
        self.subject_ids = x_train['id'].unique()

        for subject_id in self.subject_ids:

            x_train_filtered = x_train[x_train['id'] == subject_id].copy()
            subset_df = x_train_filtered[-training_samples_per_subject:].reset_index()

            # Ensure a date range with 5-minute frequency covering the full range of the DataFrame
            subset_df.set_index('t', inplace=True)
            full_range = pd.date_range(start=subset_df.index.min(), end=subset_df.index.max(), freq='5min')
            subset_df = subset_df.reindex(full_range)
            subset_df = subset_df.reset_index().rename(columns={'level_0': 't'})
            columns_to_fill = ['basal', 'bolus', 'cho']
            subset_df[columns_to_fill] = subset_df[columns_to_fill].fillna(0)

            rbg = ReplayBG(modality=modality, data=subset_df, bw=self.bw, scenario=self.scenario,
                           save_name=self.safe_file_string, save_folder=self.safe_file_string, n_steps=n_steps,
                           seed=1, plot_mode=False, verbose=True,  # Turn of when training in server
                           analyze_results=False)

            # Run identification
            rbg.run(data=subset_df, bw=self.bw)

            # Initialize some default model parameters that for some reason are commented out in ReplayBG
            rbg.model.model_parameters.ka1 = 0.0034  # 1/min (virtually 0 in 77% of the cases)

            # When using multi-meal
            # mp = rbg.model.model_parameters
            # rbg.model.model_parameters.beta = (mp.beta_B + mp.beta_L + mp.beta_D) / 3

            self.models += [rbg]

        # Delete the automatically stored draws after run
        # shutil.rmtree('results')

        return self

    def _predict_model(self, x_test, use_particle_filter=False):
        #df = self.process_input_data(x_test)
        n_pred = self.prediction_horizon // 5

        prediction_result = []
        for index, subject_id in enumerate(self.subject_ids):
            # Find parameters for the subject
            model_parameters = self.models[index].model.model_parameters
            model_parameters.ka1 = 0.0034  # 1/min (virtually 0 in 77% of the cases)

            x_test_filtered = x_test[x_test['id'] == subject_id].copy()

            for _, row in x_test_filtered.iterrows():
                # First, transform the row to a df that is at replay_bg defined format
                data = self.get_df_from_row(row).iloc[-n_pred - n_pred:].reset_index()

                # Then, use replay to get the results
                rbg = ReplayBG(modality='replay', data=data, bw=self.bw, scenario=self.scenario,
                               save_name=self.safe_file_string, save_folder=self.safe_file_string, plot_mode=False,
                               verbose=False, analyze_results=False)

                # Run it
                results = rbg.run(data=data, bw=self.bw)
                result = results['cgm']['median']

                # The simulation drifts from the CGM values (as expected), hence, we calibrate it with the measurement
                difference = row['CGM'] - result[-n_pred - 1]
                prediction = result[-n_pred:].tolist()
                prediction = [val + difference for val in prediction]
                prediction_result += [prediction]

        """
        # TODO: For this code to work, we need to add samples (with nan or 0) in each gap where there are time gaps
        # TODO: and then remove those predictions before returning result
        prediction_result = []
        for index, subject_id in enumerate(self.subject_ids):
            x_test_filtered = df[df['id'] == subject_id].copy().reset_index()

            if x_test_filtered.shape[0] == 0:
                print(f'No test samples for subject {subject_id}, proceeding to next subject...')
            else:
                model_parameters = self.models[index].model.model_parameters
                model_parameters.ka1 = 0.0034  # 1/min (virtually 0 in 77% of the cases)
                prediction_result += self.get_phy_prediction(model_parameters, x_test_filtered, self.prediction_horizon,
                                                             use_particle_filter)
        """

        # Currently, the amount of predictions are equal to samples-prediction lenght, so for 80 samples, and 72 prediction steps, we get 8 predictions
        return prediction_result

    def get_df_from_row(self, row):
        def get_dates_and_values(column, data):
            relevant_columns = [val for val in data.index if val.startswith(column)]
            dates = []
            values = []

            date = data.name
            if isinstance(date, np.int64):
                date = datetime.datetime.now()

            for col in relevant_columns:
                if col == column:
                    values.append(data[col])
                    dates.append(date)
                elif "what_if" in col:
                    values.append(data[col])
                    new_date = date + datetime.timedelta(minutes=int(col.split("_")[-1]))
                    dates.append(new_date)
                elif "label" in col:
                    pass
                else:
                    values.append(data[col])
                    new_date = date - datetime.timedelta(minutes=int(col.split("_")[-1]))
                    dates.append(new_date)

            if not dates or not values:
                # Handle the case where one or both lists are empty
                pass
            else:
                # Sorting
                combined = list(zip(dates, values))
                # Sort by the dates (first element of each tuple)
                combined.sort(key=lambda x: x[0])
                # Separate into individual lists
                dates, values = zip(*combined)

            return dates, values

        bolus_dates, bolus_values = get_dates_and_values('bolus', row)
        basal_dates, basal_values = get_dates_and_values('basal', row)
        bg_dates, bg_values = get_dates_and_values('CGM', row)
        carbs_dates, carbs_values = get_dates_and_values('carbs', row)

        date = row.name
        for time in range(5, self.prediction_horizon + 1, 5):
            bg_values = bg_values + (np.nan,)
            new_date = date + datetime.timedelta(minutes=time)
            bg_dates = bg_dates + (new_date,)

        df_bolus = pd.DataFrame({'date': bolus_dates, 'bolus': bolus_values})
        df_basal = pd.DataFrame({'date': basal_dates, 'basal': basal_values})
        df_bg = pd.DataFrame({'date': bg_dates, 'CGM': bg_values})  # CGM as glucose
        df_carbs = pd.DataFrame({'date': carbs_dates, 'carbs': carbs_values})  # carbs as cho

        df = pd.merge(df_bolus, df_basal, on='date', how='outer')
        df = pd.merge(df, df_bg, on='date', how='outer')
        df = pd.merge(df, df_carbs, on='date', how='outer')

        df.set_index('date', inplace=True)
        df = self.process_input_data(df)
        df = df.sort_values(by='t')
        return df

    def get_phy_prediction(self, model_parameters, data, prediction_horizon):
        """
        This function is translated from MatLab: https://github.com/checoisback/phy-predict/blob/main/getPhyPrediction.m.

        Computes the glucose predicted profiles by using a non-linear physiological model of glucose-insulin dynamics
        integrated into a particle filter.

        Inputs:
        - identified_model_parameters: a dictionary containing the identified model parameters described in the provided link.
        - test_data: a DataFrame containing the following columns: Time, glucose, CHO, bolus_insulin, and basal_insulin.
        The measurement units must be consistent with the ones described in the provided link.
        - PH: the prediction horizon (in minutes).
        - use_particle_filter: whether to use the deterministic physiological model, or a particle filter for online learning.

        Output:
        - prediction_results: a dictionary containing the test data (prediction_results['data']) and the predicted profile
        (prediction_results['ighat']).
        """
        Ts = 5  # minutes

        # Parameters configuration based on whether to use a particle filter or not
        n_particles = 5000  # number of particles
        sigma_v = 25  # measurements noise variance
        sigma_u0 = np.array([10, 10, 10, 0.6, 0.6, 0.6, 1e-6, 10, 10])  # process noise variance

        # set physiological model environment
        model = {'TS': 1, 'YTS': 5, 'TID': (data['t'].iloc[-1] - data['t'].iloc[0]).total_seconds() / 60 + 1}
        model['TIDSTEPS'] = int(model['TID'] / model['TS'])  # integration steps
        model['TIDYSTEPS'] = int(model['TID'] / model['YTS'])  # total identification simulation time [sample steps]
        model_parameters.TS = model['TS']

        # prepare input data
        bolus, basal, bolusDelayed, basalDelayed = self.insulin_setup_pf(data, model, model_parameters)
        cho, choDelayed = self.meal_setup_pf(data, model, model_parameters)
        meal = choDelayed
        total_ins = bolusDelayed + basalDelayed

        end_time = data['t'].iloc[0] + pd.Timedelta(minutes=len(cho) - 1)
        time_data = pd.date_range(start=data['t'].iloc[0], end=end_time, freq='min')

        # initial state
        x0 = np.array([
            0,  # Qsto1(0)
            0,  # Qsto2(0)
            0,  # Qgut(0)
            model_parameters.u2ss / (model_parameters.ka1 + model_parameters.kd),  # Isc1(0)
            (model_parameters.kd / model_parameters.ka2) * model_parameters.u2ss / (
                    model_parameters.ka1 + model_parameters.kd),  # Isc2(0)
            (model_parameters.ka1 / model_parameters.ke) * model_parameters.u2ss / (
                    model_parameters.ka1 + model_parameters.kd) +
            (model_parameters.ka2 / model_parameters.ke) * (model_parameters.kd / model_parameters.ka2) *
            model_parameters.u2ss / (model_parameters.ka1 + model_parameters.kd),  # Ip(0)
            model_parameters.Xpb,  # X(0)
            data['glucose'].iloc[0],  # G(0)
            data['glucose'].iloc[0]  # IG(0)
        ])

        sigma0 = np.eye(len(x0))

        # Set state bounds
        state_bound = np.ones((len(x0), 2))
        state_bound[:, 0] = np.array(x0) - 0.03 * np.array(x0)  # Lower bound
        state_bound[:, 1] = np.array(x0) + 0.03 * np.array(x0)  # Upper bound

        # Initialize the particle filter
        pf_v0 = ParticleFilter(gi_particle_filter_state_function, gi_measurement_likelihood_function, n_particles, x0,
                               np.diag(sigma0))

        _, _, _, IG_hat, _, _ = self.apply_pf(pf_v0, time_data, data['glucose'], meal, total_ins, x0, sigma_u0,
                                              sigma_v, model_parameters, prediction_horizon)

        # Get all the predicted values in 5-minute intervals
        predicted_trajectories = IG_hat[:, Ts - 1::Ts]

        # Convert all zeros in a sublist to np.nan if ALL are zeros
        predicted_trajectories = [
            [np.nan if num == 0 else num for num in sublist] if all(num == 0 for num in sublist) else sublist.tolist()
            for sublist in predicted_trajectories
        ]

        # TODO: Use what-if events to add samples in the end to get all predictions
        # TODO: Make an opportunity to choose between what-if and agnostic predictions
        # What-if predictions can be achieved by imputing the next cho and insulin state directly in the state
        # transition function. The predictions here are step wise / recursive, and given the (estimate) of the
        # previous state. Hence, we could just use the true state instead.

        return predicted_trajectories

    def insulin_setup_pf(self, data, model, model_parameters):
        """
        Generates the vectors containing the insulin infusions to be used to simulate the physiological model.

        Inputs:
            - data: a DataFrame containing the data to be used by the tool;
            - model: a dictionary that contains general parameters of the physiological model;
            - model_parameters: a dictionary containing the model parameters.
        Outputs:
            - bolus: a vector containing the insulin bolus dose at each time step [mU/min*kg];
            - basal: a vector containing the basal insulin value at each time step [mU/min*kg];
            - bolusDelayed: a vector containing the insulin bolus dose at each time step delayed by tau min [mU/min*kg];
            - basalDelayed: a vector containing the basal insulin value at each time step delayed by tau min [mU/min*kg].
        """
        # Initialize the basal and bolus vectors
        basal = np.zeros(int(model['TIDSTEPS']))
        bolus = np.zeros(int(model['TIDSTEPS']))

        # Set the basal and bolus vectors
        for time in range(len(np.arange(0, model['TID'], model['YTS']))):
            start_index = int(1 + time * (model['YTS'] / model['TS']) - 1)
            end_index = int((time + 1) * (model['YTS'] / model['TS']))
            bolus[start_index:end_index] = data['bolus'].iloc[time] * 1000 / model_parameters.bw  # mU/(kg*min)
            basal[start_index:end_index] = data['basal'].iloc[time] * 1000 / model_parameters.bw  # mU/(kg*min)

        # Add delay in insulin absorption
        bolus_delay = int(np.floor(model_parameters.tau / model['TS']))
        bolus_delay = int(np.round(bolus_delay / 5) * 5)
        bolus_delayed = np.concatenate((np.zeros(bolus_delay), bolus))
        bolus_delayed = bolus_delayed[:int(model['TIDSTEPS'])]

        basal_delayed = np.concatenate((np.ones(bolus_delay) * basal[0], basal))
        basal_delayed = basal_delayed[:int(model['TIDSTEPS'])]

        basal = basal[:int(model['TIDSTEPS'])]
        bolus = bolus[:int(model['TIDSTEPS'])]

        return bolus, basal, bolus_delayed, basal_delayed

    def meal_setup_pf(self, data, model, model_parameters):
        """
        Generates the vector containing the CHO intake events to be used to simulate the physiological model.

        Inputs:
            - data: a DataFrame containing the data to be used by the tool;
            - model: a dictionary that contains general parameters of the physiological model;
            - model_parameters: a dictionary containing the model parameters;
            - environment: a dictionary that contains general parameters to be used by ReplayBG.
        Outputs:
            - meal: a vector containing the carbohydrate intake at each time step [mg/min*kg];
            - mealDelayed: a vector containing the carbohydrate intake at each time step delayed by beta min [mg/min*kg].
        """
        # Initialize the meal vector
        meal = np.zeros(model['TIDSTEPS'])

        # Set the meal vector
        for time in range(len(np.arange(0, model['TID'], model['YTS']))):
            start_index = int(1 + time * (model['YTS'] / model['TS']) - 1)
            end_index = int((time + 1) * (model['YTS'] / model['TS']))
            meal[start_index:end_index] = data['cho'][time] * 1000 / model_parameters.bw

        # Add delay of main meal absorption
        meal_delay = int(round(model_parameters.beta / model['TS']))
        meal_delay = int(round(meal_delay / 5) * 5)
        meal_delayed = np.concatenate((np.zeros(meal_delay), meal))
        meal_delayed = meal_delayed[:model['TIDSTEPS']]
        meal = meal[:model['TIDSTEPS']]

        return meal, meal_delayed

    def apply_pf(self, pf, time, noisy_measure, meal, total_ins, x0, sigma_u0, sigma_v, model_parameters,
                 prediction_horizon):
        """
        Applies a particle filter to perform real-time filtering.

        Inputs:
            - pf: a particle filter object
            - time: array containing the time information
            - noisy_measure: array containing the noisy measurements
            - meal: array containing the carbohydrate intake at each time step [mg/min*kg]
            - total_ins: array containing the total insulin intake at each time step [mU/min*kg]
            - kwargs: additional keyword arguments including:
                - InitialCondition_x0: initial state condition
                - InitialCondition_sigma0: initial state covariance
                - processNoiseVariance: process noise variance
                - measurementNoiseVariance: measurement noise variance
                - parameterStructure: structure containing model parameters
                - predictionHorizon: prediction horizon (default: 30)

        Outputs:
            - lastBestGuess: matrix containing the last best guess of state variables
            - lastBestCov: matrix containing the last best covariance of state variables
            - G_hat: matrix containing the predicted glucose values
            - IG_hat: matrix containing the predicted interstitial glucose values
            - VarG_hat: matrix containing the predicted variance of glucose
            - VarIG_hat: matrix containing the predicted variance of interstitial glucose
        """
        sigma_u = np.sqrt(sigma_u0)
        sigma_v = np.sqrt(sigma_v)

        lastBestGuess = np.zeros((len(x0), len(noisy_measure)))
        lastBestGuess[7, 0] = noisy_measure[0]  # Initial glucose
        lastBestGuess[8, 0] = noisy_measure[0]  # Initial interstitial glucose
        lastBestCov = np.zeros((len(x0), len(noisy_measure)))

        IG_hat = np.zeros((len(noisy_measure), prediction_horizon))
        G_hat = np.zeros((len(noisy_measure), prediction_horizon))
        VarIG_hat = np.zeros((len(noisy_measure), prediction_horizon))
        VarG_hat = np.zeros((len(noisy_measure), prediction_horizon))

        print_progress_bar(1, len(time), prefix='Progress:', suffix='Complete', length=50)

        for k in range(len(time)):
            if k % 50 == 0:
                print_progress_bar(k + 1, len(time), prefix='Progress:', suffix='Complete', length=50)

            # Prediction step
            state_pred, cov_pred = pf.predict(meal[k], total_ins[k], time[k].hour, sigma_u, model_parameters)

            index_measure = int(k / (5 / model_parameters.TS))
            # Measurement checking based on index
            if k % int(5 / model_parameters.TS) == 0:
                CGM_measure = noisy_measure[index_measure]
            else:
                CGM_measure = np.nan

            # Correction step
            if not np.isnan(CGM_measure):
                state_corrected, cov_corrected = pf.correct(CGM_measure, sigma_v)
            else:
                state_corrected = state_pred
                cov_corrected = cov_pred

            # Saving the last best guess and covariance
            if (k % int(5 / model_parameters.TS)) == 0:
                lastBestGuess[:, index_measure] = state_corrected[:len(x0)]
                lastBestCov[:, index_measure] = np.diag(cov_corrected)

            # k-step ahead prediction, every interval of a given sampling rate for the output
            if (k + prediction_horizon <= len(time)) and ((k % int(5 / model_parameters.TS)) == 0):
                pf_pred = copy.deepcopy(pf)
                mP_pred = copy.deepcopy(model_parameters)

                for p in range(prediction_horizon):
                    # This function assumes that future meals and insulin injections are known
                    # Alternatively, we could use estimates or assumptions of what these future values probably will be
                    step_ahead_prediction, cov_ahead_prediction = pf_pred.predict(meal[k + p], total_ins[k + p],
                                                                                  time[k + p].hour, sigma_u, mP_pred)

                    G_hat[index_measure, p] = step_ahead_prediction[7]  # Glucose
                    IG_hat[index_measure, p] = step_ahead_prediction[8]  # Interstitial glucose

                    pred_variance = np.diag(cov_ahead_prediction)
                    VarG_hat[index_measure, p] = pred_variance[7]  # Variance for glucose
                    VarIG_hat[index_measure, p] = pred_variance[8]  # Variance for interstitial glucose

        return lastBestGuess, lastBestCov, G_hat, IG_hat, VarG_hat, VarIG_hat

    def best_params(self):
        #self.print_model_parameters()
        return

    def process_input_data(self, x_df):
        processed_df = x_df.copy()
        processed_df['glucose'] = x_df['CGM']
        processed_df['cho'] = x_df['carbs']
        processed_df['CR'] = np.nan
        processed_df['CF'] = np.nan
        processed_df['cho'] = processed_df['cho'] / 5  # Carbs needs to be in grams/min and not grams
        processed_df['bolus'] = processed_df['bolus'] / 5  # Bolus needs to be in U/min and not U
        processed_df['basal'] = processed_df['basal'] / 60  # Basal needs to be in U/min and not U/hr
        processed_df['cho_label'] = np.nan
        processed_df['bolus_label'] = np.nan
        processed_df['exercise'] = 0
        processed_df['t'] = processed_df.index
        processed_df.reset_index(inplace=True)

        def assign_label(row):
            time_of_day = row['t'].hour
            if time_of_day < 4 or time_of_day >= 17:
                return 'D' # Dinner label
            elif 4 <= time_of_day < 11:
                return 'B' # Breakfast label
            else:
                return 'L' # Lunch label

        processed_df['cho_label'] = processed_df['cho_label'].astype(object)
        processed_df['bolus_label'] = processed_df['bolus_label'].astype(object)
        processed_df.loc[processed_df['cho'] > 0, 'cho_label'] = processed_df[processed_df['cho'] > 0].apply(assign_label, axis=1)
        processed_df.loc[processed_df['bolus'] > 0, 'bolus_label'] = processed_df[processed_df['bolus'] > 0].apply(assign_label, axis=1)

        return processed_df

    def process_data(self, df, model_config_manager, real_time):
        # df = df.dropna()

        df = process_data(df, model_config_manager, real_time)

        # Fill nan values with 0 as per instructions: https://github.com/gcappon/py_replay_bg/tree/master/docs/src/data_requirements
        columns_to_fill = ['basal', 'bolus', 'carbs']
        df[columns_to_fill] = df[columns_to_fill].fillna(0)

        return df

    def print_model_parameters(self):
        for i in range(len(self.models)):
            mp = self.models[i].model.model_parameters

            if self.scenario == 'multi-meal':
                print("beta_B", mp.beta_B)  # Time delays for meal Breakfast, Lunch and Dinner
                print("beta_L", mp.beta_L)
                print("beta_D", mp.beta_D)

                print("kabs_D", mp.kabs_D)  # Absorption specific kinetics
                print("kabs_B", mp.kabs_B)
                print("kabs_L", mp.kabs_L)

                print("SI_D", mp.SI_D)  # Insulin sensitivity indexes
                print("SI_B", mp.SI_B)
                print("SI_L", mp.SI_L)
            else:
                print("beta", mp.beta)
                print("kabs", mp.kabs)
                print("SI", mp.SI)

            print("bw:", mp.bw)  # Body weight
            print("tau:", mp.tau)  # General time delay
            print("u2ss:", mp.u2ss)  # Basal insulin rate
            print("ka1:", mp.ka1)  # Absorption Rates
            print("ka2:", mp.ka2)
            print("kd:", mp.kd)  # Degradation Rate, the rate at which a substance degrades or is cleared from the system
            print("Xpb:", mp.Xpb)  # Initial insulin action?
            print("Gb:", mp.Gb)  # Baseline glucose level
            print("r2:", mp.r2)  # Parameter in Risk Model or Regression Coefficient
            print("ke:", mp.ke)  # Elimination rate constant, how quickly a substance is removed from the bloodstream
            print("kgri:", mp.kgri)  # The rate of gastric emptying
            print("kempt:", mp.kempt)  # The rate of gastric emptying
            print("f:", mp.f)  # Absorption fraction or bioavailability
            print("p2:", mp.p2)  # Insulin sensitivity parameter
            print("VI:", mp.VI)  # Volume of insulin distribution
            print("SG:", mp.SG)  # Glucose sensitivity factor
            print("VG:", mp.VG)  # Volume of glucose distribution
            print("alpha:", mp.alpha)  # Rate constant or conversion factor


def gi_measurement_likelihood_function(predicted_measurement, measurement, sigma_v):
    # The measurement is the ninth state. Extract all measurement hypotheses from particles
    # Calculate the likelihood of each predicted measurement

    likelihood = norm.pdf(predicted_measurement, measurement, sigma_v)

    if likelihood < 1e-300:
        return 1e-300  # Avoid returning zero, which would break the particle filter

    return likelihood


def gi_particle_filter_state_function(particles, CHO, INS, time, sigma_u, mP):
    numberOfParticles, numberOfStates = particles.shape

    # Time-propagate each particle using Euler integration
    dt = mP.TS  # Sample time
    for kk in range(numberOfParticles):
        particles[kk, :] = particles[kk, :] + gi_state_function_continuous(particles[kk, :], CHO, INS, time, mP) * dt

    # Add Gaussian noise with specified variance processNoise
    process_noise = np.diag(sigma_u)
    noise = np.dot(process_noise, np.random.randn(numberOfStates, numberOfParticles)).T  # Transpose to fit particles
    particles += noise

    return particles


def gi_state_function_continuous(x, CHO, INS, time, mP):
    dxdt = np.zeros_like(x)
    Qsto1, Qsto2, Qgut, Isc1, Isc2, Ip, X, G, IG = x

    # Compute the basal plasmatic insulin
    Ipb = (mP.ka1 / mP.ke) * (mP.u2ss) / (mP.ka1 + mP.kd) + (mP.ka2 / mP.ke) * (mP.kd / mP.ka2) * (mP.u2ss) / (
            mP.ka1 + mP.kd)

    if hasattr(mP, 'SI'):
        SI = mP.SI
        kabs = mP.kabs
    else:
        # If multi-meal
        # Calculate state derivatives
        if time < 4 or time >= 17:
            SI = mP.SI_D
            kabs = mP.kabs_D
        elif 4 <= time < 11:
            SI = mP.SI_B
            kabs = mP.kabs_B
        else:
            SI = mP.SI_L
            kabs = mP.kabs_L

    risk = compute_hypoglycemic_risk(x[7], mP)  # Assuming G is x[7]

    rhoRisk = 1 + risk
    Ra = mP.f * kabs * Qgut
    dxdt[0] = -mP.kgri * Qsto1 + CHO
    dxdt[1] = mP.kgri * Qsto1 - mP.kempt * Qsto2
    dxdt[2] = mP.kempt * Qsto2 - kabs * Qgut
    dxdt[3] = -mP.kd * Isc1 + INS
    dxdt[4] = mP.kd * Isc1 - mP.ka2 * Isc2
    dxdt[5] = mP.ka2 * Isc2 - mP.ke * Ip
    dxdt[6] = -mP.p2 * (X - (SI / mP.VI) * (Ip - Ipb))
    dxdt[7] = -((mP.SG + rhoRisk * X) * G) + mP.SG * mP.Gb + Ra / mP.VG
    dxdt[8] = -(1 / mP.alpha) * (IG - G)

    return dxdt


def compute_hypoglycemic_risk(G, mP):
    # Function to compute hypoglycemic risk as described in Visentin et al., JDST, 2018.

    # Setting the risk model threshold
    Gth = 60

    # For all values below the risk model threshold, the hypoglycemic risk will be the same.
    # This redefinition of G avoid some output errors.
    if G < Gth:
        G = Gth - 1

    # Compute the risk
    Gb = mP.Gb
    risk = (10 * (np.log(G) ** mP.r2 - np.log(Gb) ** mP.r2) ** 2 * ((G < Gb) & (G >= Gth)) +
            10 * (np.log(Gth) ** mP.r2 - np.log(Gb) ** mP.r2) ** 2 * (G < Gth))

    return abs(risk)


class ParticleFilter:
    def __init__(self, state_transition_fn, measurement_fn, num_particles, x0, sigma0):
        self.num_particles = num_particles

        # Set state bounds
        lower_bound = np.array(x0) - 0.03 * np.array(x0)  # Lower bound
        upper_bound = np.array(x0) + 0.03 * np.array(x0)  # Upper bound

        # Initialize particles uniformly within the bounds
        self.particles = np.random.uniform(low=lower_bound, high=upper_bound, size=(num_particles, len(x0)))
        self.weights = np.ones(num_particles) / num_particles
        self.state_transition_fn = state_transition_fn  # gi_particle_filter_state_function
        self.measurement_fn = measurement_fn  # gi_measurement_likelihood_function


    def predict(self, carbs, insulin, time, sigma_u, model_parameters):
        """Predict the next state of the particles."""

        self.particles = self.state_transition_fn(self.particles, carbs, insulin, time, sigma_u, model_parameters)

        # Calculate the mean of the particles
        mean_state = np.mean(self.particles, axis=0)

        # Calculate the covariance matrix of the particles
        cov_state = np.cov(self.particles, rowvar=False)  # rowvar=False to treat rows as variables

        return mean_state, cov_state

    def correct(self, measurement, sigma_v):
        """
        Correct/update the particle weights based on the measurement, and resample.

        Parameters:
            measurement (float): The new measurement.
            sigma_v (float): Measurement noise standard deviation.

        Returns:
            tuple: Corrected state estimate and its covariance.
        """
        # Update weights based on measurement likelihood
        for i in range(self.num_particles):
            expected_measurement = self.measurement_fn(self.particles[i][8], measurement, sigma_v)
            self.weights[i] *= expected_measurement

        # Normalize weights
        self.weights += 1.e-300  # avoid division by zero
        self.weights /= np.sum(self.weights)

        # Resample particles to avoid degeneracy
        indexes = self.resample_particles()
        self.particles = self.particles[indexes]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights after resampling

        # Calculate the new state estimate and covariance
        mean_state = np.mean(self.particles, axis=0)
        cov_state = np.cov(self.particles, rowvar=False)

        return mean_state, cov_state

    def resample_particles(self):
        """Resample particles proportionally to their weight."""
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # ensure sum is exactly one
        return np.searchsorted(cumulative_sum, np.random.random(self.num_particles))


class MockModelParameters:
    def __init__(self):
        self.bw = 100
        self.beta_B = 1.9210921467522486
        self.beta_L = 7.227110102827593
        self.beta_D = 22.740130507858265
        self.beta = (self.beta_B + self.beta_L + self.beta_D) / 3
        self.tau = 8
        self.u2ss = 0.1291666666666666
        self.ka1 = 0.0034
        self.ka2 = 0.0004536788602030393
        self.kd = 0.09433329008933948
        self.kabs_D = 0.02161163589225169
        self.kabs_B = 0.020427660145477033
        self.kabs_L = 0.0012429702583311433
        self.Xpb = 0.0
        self.SI_D = 0.0007412325384855292
        self.SI_B = 0.0007072567151272324
        self.SI_L = 0.0007701611742637199
        self.Gb = 140.98122622460085
        self.r2 = 0.8124
        self.ke = 0.127
        self.kgri = 0.28500571715267026
        self.kempt = 0.28500571715267026
        self.f = 0.9
        self.p2 = 0.012
        self.VI = 0.126
        self.SG = 0.018425998653797095
        self.VG = 1.45
        self.alpha = 7
        self.TS = None


def print_progress_bar(iteration, total, prefix='Progress:', suffix='Complete', decimals=1, length=50, fill='█',
                       print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()
