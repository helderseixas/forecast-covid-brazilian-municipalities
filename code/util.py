import numpy as np
from scipy.stats import gamma
import epyestim
import epyestim.covid19 as covid19
# plt.style.use('seaborn-white')
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns

default_line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]

colors_default = {'extremamente_maior': '#80191C', 'muito_maior': '#D7191C', 'maior': '#FDAE61', 'medio': '#F09CFA',
                  'menor': '#ABD9E9', 'muito_menor': '#2c7bb6'}

def get_default_colors_maps(n):
    colors_custom = [colors_default['muito_menor'], colors_default['menor'], colors_default['medio'],
                     colors_default['maior'], colors_default['muito_maior'], colors_default['extremamente_maior']]
    pallet_custom = ListedColormap(colors_custom[:n], name='map')
    return pallet_custom

def get_default_colors_divergence_seaborn(n, reverse=False):
    colors_custom = [colors_default['muito_menor'], colors_default['menor'], colors_default['medio'],
                     colors_default['maior'], colors_default['muito_maior'], colors_default['extremamente_maior']]

    if reverse == True:
        pallet_custom = sns.color_palette(list(reversed(colors_custom[:n])))
    else:
        pallet_custom = sns.color_palette(colors_custom[:n])
    return pallet_custom

def get_default_colors_divergence_seaborn_set_2(n, reverse=False):
    colors_custom = [colors_default['muito_menor'], '#91E693', colors_default['medio'],
                     colors_default['maior'], colors_default['muito_maior'], colors_default['extremamente_maior']]

    if reverse == True:
        pallet_custom = sns.color_palette(list(reversed(colors_custom[:n])))
    else:
        pallet_custom = sns.color_palette(colors_custom[:n])
    return pallet_custom

def get_default_colors_categorical_seaborn(n=5):
    colors_custom = ['#FEC4DC', '#3F80B3', '#610099', '#CCA86C', '#91E693']
    pallet_custom = sns.color_palette(colors_custom[:n])
    return pallet_custom

def get_default_colors_temporal_series_highlighting_peaks_seaborn(n=5):
    colors_custom = ['#C7C7C7', '#FF2B2B', '#3331E6', '#FFB60F', '#BF00F5']
    pallet_custom = sns.color_palette(colors_custom[:n])
    return pallet_custom

def get_default_colors_heatmap():
    colors = [[0, 'darkblue'],
              [0.45, '#F0F0F0'],
              [0.55, '#F0F0F0'],
              [1, 'darkred']]
    return LinearSegmentedColormap.from_list('', colors)

def estimate_effective_reproduction_number(df_cases, smoothing_window = 28, r_window_size = 14):
    # Flaxman, Seth, et al. "Estimating the effects of non-pharmaceutical interventions on COVID-19 in Europe." Nature 584.7820 (2020): 257-261.
    # Gamma distribution with serial interval mean equal to 4.03 days, gamma(6.5, scale=0.62)
    standard_serial_inteval_distribution = covid19.generate_standard_si_distribution()

    # Ganyani, Tapiwa, et al. "Estimating the generation interval for coronavirus disease (COVID-19) based on symptom onset data, March 2020." Eurosurveillance 25.17 (2020): 2000257.
    # Gamma distribution with incabation mean equal to 5.22 days.
    incubation_time_distribution = epyestim.discrete_distrb(gamma(a=3.45, scale=1 / 0.66))

    # Chagas, Eduarda TC, et al. "Effects of population mobility_for_hsi on the COVID-19 spread in Brazil." PloS one 16.12 (2021): e0260610.
    # Exponential distribution with delay mean equal to 10.85 days
    delay_onset_to_notification_distribution = epyestim.discrete_distrb(gamma(a=1, scale=10.85))

    # Distribution resultant of convolution incubation_time_distribution and delay_onset_to_notification_distribution.
    # Represent delay of infection to notification with delay mean equal to 16.07 days.
    delay_infecton_to_notification_distribution = np.convolve(incubation_time_distribution,
                                                              delay_onset_to_notification_distribution)

    # Cori, A., Ferguson, N. M., Fraser, C., & Cauchemez, S. (2013). A new framework and software to estimate time-varying reproduction numbers during epidemics. American journal of epidemiology, 178(9), 1505-1512.
    # https://github.com/lo-hfk/epyestim
    df_effective_reproduction_number = covid19.r_covid(df_cases, smoothing_window = smoothing_window, r_window_size = r_window_size, auto_cutoff=True, n_samples=10, delay_distribution=delay_infecton_to_notification_distribution, gt_distribution=standard_serial_inteval_distribution)
    df_effective_reproduction_number = df_effective_reproduction_number.reset_index()
    df_effective_reproduction_number = df_effective_reproduction_number.rename(
        columns={'index': 'DATA', 'R_mean': 'NUMERO_REPRODUCAO_EFETIVO_MEDIA',
                 'R_var': 'NUMERO_REPRODUCAO_EFETIVO_VARIANCIA', 'Q0.025': 'NUMERO_REPRODUCAO_EFETIVO_QUANTIL_0.025',
                 'Q0.5': 'NUMERO_REPRODUCAO_EFETIVO_MEDIANA', 'Q0.975': 'NUMERO_REPRODUCAO_EFETIVO_QUANTIL_0.975'})
    df_effective_reproduction_number = df_effective_reproduction_number.drop(columns=['cases'])

    return df_effective_reproduction_number

def estimate_daily_growth_rate(new_cases):
    return estimate_t_days_growth_rate(new_cases, 1)

def estimate_14_days_growth_rate(new_cases):
    return estimate_t_days_growth_rate(new_cases, 14)

def estimate_t_days_growth_rate(new_cases, t):
    gr = np.subtract(new_cases[t:], new_cases[:len(new_cases) - t])
    gr = np.divide(gr, new_cases[:len(new_cases) - t])
    return gr

def calculating_empirical_14_days_reproductive_ratio(new_cases):
    re = (new_cases[14:]) / (new_cases[:len(new_cases) - 14])
    return re

def get_days_to_ignore_extreme_subnotification_period(case_evolution_data, dates, valley_1):
    mean_case_evolution_first_wave = np.nanmean(case_evolution_data[(dates <= valley_1)])
    max_case_evolution_first_wave = np.nanmax(case_evolution_data[(dates <= valley_1)])
    date_max_case_evolution_first_wave = dates[case_evolution_data == max_case_evolution_first_wave][0]
    print('date_max_cases_first_wave: '+str(date_max_case_evolution_first_wave))
    date_after_extreme_subnotification_period = dates[(dates > date_max_case_evolution_first_wave) & (dates <= valley_1) & (case_evolution_data < mean_case_evolution_first_wave)][0]
    return int((date_after_extreme_subnotification_period - dates[0]) / np.timedelta64(1, 'D'))

def centimeter_to_inch(centimeters):
    return centimeters * 1/2.54

def calculate_confidence_interval(samples):
    # Number of bootstrap resamples
    num_resamples = 1000

    # Create an array to store resampled means
    resample_means = np.zeros(num_resamples)

    # Perform bootstrapping
    for i in range(num_resamples):
        resample = np.random.choice(samples, size=len(samples), replace=True)
        resample_means[i] = np.mean(resample)

    # Calculate confidence interval
    lower_bound = np.percentile(resample_means, 2.5)
    upper_bound = np.percentile(resample_means, 97.5)

    return lower_bound, upper_bound

def calculate_confidence_interval_matrix(matrix):
    confidence_intervals = []
    for j in range(matrix.shape[1]):
        samples = matrix[:,j]
        lower_bound, upper_bound = calculate_confidence_interval(samples)
        confidence_intervals.append([lower_bound, upper_bound])
    return confidence_intervals
