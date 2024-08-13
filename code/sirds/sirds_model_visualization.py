import ast
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from PIL import Image

from code import util
from sirds_model import sirds_model_fuzzy, get_new_deaths, get_new_cases, \
    get_epidemic_periods_for_beta_transition_fuzzy_variable, get_epidemic_periods_with_slow_transition_fuzzy_variable, \
    get_fuzzy_effective_reproduction_number, epidemic_parameter_defuzzification, get_error_deaths_rt
import matplotlib.lines as mlines


def get_sirds(result, forecast_horizon=0, fatality_in_forecast_period=None, contact_rate_in_forecast_period=None,
              loss_immunity_rate_in_forecast_period=None):
    x_days_between_infections = result.filter(like='x_days_between_infections').values.tolist()[1:]
    x_case_fatality_probability = result.filter(like='x_case_fatality_probability').values.tolist()
    x_loss_immunity_in_days = result.filter(like='x_loss_immunity_in_days').values.tolist()
    x_breakpoint = result.filter(like='x_breakpoint').values.tolist()
    x_transition_days_between_epidemic_periods = result.filter(
        like='x_transition_days_between_epidemic_periods').values.tolist()
    list_breakpoints_in_slow_transition = ast.literal_eval(result.list_breakpoints_in_slow_transition)

    y = sirds_model_fuzzy(tuple([*[
        result.x_initial_infected_population,
        result.period_in_days,
        forecast_horizon,
        fatality_in_forecast_period,
        contact_rate_in_forecast_period,
        loss_immunity_rate_in_forecast_period,
        result.x_days_between_infections_0,
        result.days_to_recovery, list_breakpoints_in_slow_transition], *x_case_fatality_probability,
                                 *x_loss_immunity_in_days, *x_days_between_infections, *x_breakpoint,
                                 *x_transition_days_between_epidemic_periods]))

    return y


def get_sirds_extras(result, S, D, I_accumulated, forecast_horizon=None, estimated_fatality_in_forecast_period=None,
                     estimated_days_between_infections_in_forecast_period=None,
                     loss_immunity_in_day_in_forecast_period=None):
    fitted_period_in_days = int(result.period_in_days)
    if forecast_horizon is not None:
        period_in_days = fitted_period_in_days + forecast_horizon

    D_new_deaths = get_new_deaths(D)
    I_new_cases = get_new_cases(I_accumulated)

    x_days_between_infections = result.filter(like='x_days_between_infections').values.tolist()[1:]
    x_breakpoint = result.filter(like='x_breakpoint').values.tolist()
    x_case_fatality_probability = result.filter(like='x_case_fatality_probability').values.tolist()
    x_loss_immunity_in_days = result.filter(like='x_loss_immunity_in_days').values.tolist()
    x_transition_days_between_epidemic_periods = result.filter(
        like='x_transition_days_between_epidemic_periods').values.tolist()

    days_between_infections_values_full = np.append([result.x_days_between_infections_0], x_days_between_infections)

    epidemic_periods_with_fast_transition_fuzzy_variable = get_epidemic_periods_for_beta_transition_fuzzy_variable(
        period_in_days, x_breakpoint, x_transition_days_between_epidemic_periods)

    list_breakpoints_in_slow_transition = ast.literal_eval(result.list_breakpoints_in_slow_transition)
    slow_transition_breakpoint_values = []
    for indice_breakpoint in list_breakpoints_in_slow_transition:
        slow_transition_breakpoint_values.append(x_breakpoint[indice_breakpoint])

    epidemic_periods_with_slow_transition_fuzzy_variable = get_epidemic_periods_with_slow_transition_fuzzy_variable(
        period_in_days, slow_transition_breakpoint_values)

    SIRDS_effective_reproduction_number = get_fuzzy_effective_reproduction_number(S, 100000, tuple([*[
        result.days_to_recovery,
        (epidemic_periods_with_fast_transition_fuzzy_variable, days_between_infections_values_full)]]))

    estimated_days_between_infections = np.array([])
    for i in range(period_in_days):
        estimation = epidemic_parameter_defuzzification(epidemic_periods_with_fast_transition_fuzzy_variable,
                                                  days_between_infections_values_full,
                                                        i,
                                                        fit_period_in_days=fitted_period_in_days,
                                                        epidemic_parameter_in_forecast_period=estimated_days_between_infections_in_forecast_period)
        estimated_days_between_infections = np.append(estimated_days_between_infections, [estimation])

    estimated_case_fatality_probability = np.array([])
    for i in range(period_in_days):
        estimation = epidemic_parameter_defuzzification(epidemic_periods_with_slow_transition_fuzzy_variable,
                                                  x_case_fatality_probability,
                                                        i,
                                                        fit_period_in_days=fitted_period_in_days,
                                                        epidemic_parameter_in_forecast_period=estimated_fatality_in_forecast_period)
        estimated_case_fatality_probability = np.append(estimated_case_fatality_probability, [estimation])

    estimated_loss_immunity_in_days = np.array([])
    for i in range(period_in_days):
        estimation = epidemic_parameter_defuzzification(epidemic_periods_with_slow_transition_fuzzy_variable,
                                                  x_loss_immunity_in_days,
                                                        i,
                                                        fit_period_in_days=fitted_period_in_days,
                                                        epidemic_parameter_in_forecast_period=loss_immunity_in_day_in_forecast_period)
        estimated_loss_immunity_in_days = np.append(estimated_loss_immunity_in_days, [estimation])

    return (D_new_deaths,
            SIRDS_effective_reproduction_number,
            I_new_cases,
            epidemic_periods_with_fast_transition_fuzzy_variable,
            epidemic_periods_with_slow_transition_fuzzy_variable,
            days_between_infections_values_full,
            x_case_fatality_probability,
            x_loss_immunity_in_days,
            estimated_days_between_infections,
            estimated_case_fatality_probability,
            estimated_loss_immunity_in_days)


def _show_performance(real_new_deaths, D_new_deaths, real_reproduction_number, reproduction_number_sird):
    mae = get_error_deaths_rt(D_new_deaths,
                        real_new_deaths,
                        reproduction_number_sird,
                        real_reproduction_number)

    print('\nGeneral MAE: ' + str(round(mae, 3)))

    print('\nNew deaths:')
    sse = mean_squared_error(D_new_deaths, real_new_deaths)
    r2 = r2_score(D_new_deaths, real_new_deaths)
    print('SSE: ' + str(round(sse, 3)))
    print('r2: ' + str(round(r2, 3)))

    print('\nRt:')
    indices_to_remove = np.argwhere(np.isnan(real_reproduction_number))
    real_reproduction_number = np.delete(real_reproduction_number, indices_to_remove)
    reproduction_number_sird_train = np.delete(reproduction_number_sird, indices_to_remove)
    sse = mean_squared_error(real_reproduction_number, reproduction_number_sird_train)
    print('SSE: ' + str(round(sse, 3)))
    r2 = r2_score(real_reproduction_number, reproduction_number_sird_train)
    print('r2: ' + str(round(r2, 3)))

def show_performance_single(real_new_deaths, D_new_deaths, real_reproduction_number, reproduction_number_sird, train_period=None):
    if train_period is None:
        train_period = len(real_new_deaths)

    D_fitted = D_new_deaths[:train_period-1]
    real_new_deaths_fitted = real_new_deaths[1:train_period]
    reproduction_number_sird_fitted = reproduction_number_sird[:train_period]
    real_reproduction_number_fitted = real_reproduction_number[:train_period]
    print('\n***Fit***')
    _show_performance(real_new_deaths_fitted, D_fitted, real_reproduction_number_fitted, reproduction_number_sird_fitted)

    if train_period < len(real_new_deaths):
        D_predicted = D_new_deaths[train_period-1:]
        real_new_deaths_predicted = real_new_deaths[train_period:]
        reproduction_number_sird_predicted = reproduction_number_sird[train_period:]
        real_reproduction_number_predicted = real_reproduction_number[train_period:]
        print('\n***Forecasting***')
        _show_performance(real_new_deaths_predicted, D_predicted, real_reproduction_number_predicted,
                      reproduction_number_sird_predicted)

def show_performance(dict_performance):
    for measurement, values in dict_performance.items():
        mean = np.mean(values)
        lower_bound, upper_bound = util.calculate_confidence_interval(values)
        print(measurement,': ', mean, '(', lower_bound, ',', upper_bound, ')')

def plot_result_single(y, D_new_deaths, real_new_deaths, real_total_deaths, real_reproduction_number, reproduction_number_sird,
                real_total_cases, real_new_cases, I_new_cases, dates,
                epidemic_periods_with_fast_transition_fuzzy_variable,
                epidemic_periods_with_slow_transition_fuzzy_variable, days_between_infections_values, days_to_recovery,
                case_fatality_probability_values, loss_immunity_in_days_values, save, directory_to_save='images',
                id_in_file='', max_date_to_fit=None):
    S, I, R, D, I_accumulated = y
    min_length = min(len(real_new_deaths), len(D))
    S = S[:min_length]
    I = I[:min_length]
    R = R[:min_length]
    D = D[:min_length]
    I_accumulated = I_accumulated[:min_length]

    mask_date = mdates.DateFormatter('%m/%Y')
    line_styles = ['-', '--', ':', '-.', '-']
    sns.set(font_scale=1.1)
    sns.set_style("ticks")
    sns.set_palette(util.get_default_colors_categorical_seaborn())
    fig, ax = plt.subplots(3, 2, figsize=(util.centimeter_to_inch(34.8), util.centimeter_to_inch(26.1)), sharex=False)

    # Plot the data on three separate curves for S(t), I(t), R(t) and D(t)
    sns.lineplot(x=dates, y=S, label='Susceptible', color=util.get_default_colors_categorical_seaborn()[1], legend=True,
                 linestyle=line_styles[0], ax=ax.flatten()[0])
    sns.lineplot(x=dates, y=I, label='Infected', color=util.get_default_colors_categorical_seaborn()[2], legend=True,
                 linestyle=line_styles[1], ax=ax.flatten()[0])
    sns.lineplot(x=dates, y=R, label='Recovered', color=util.get_default_colors_categorical_seaborn()[4], legend=True,
                 linestyle=line_styles[2], ax=ax.flatten()[0])
    sns.lineplot(x=dates, y=D, label='Deceased', color=util.get_default_colors_categorical_seaborn()[3], legend=True,
                 linestyle=line_styles[3], ax=ax.flatten()[0])
    if max_date_to_fit is not None:
        ax.flatten()[0].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[0].set_xlabel('Month/Year')
    ax.flatten()[0].xaxis.set_major_formatter(mask_date)
    ax.flatten()[0].tick_params(axis='x', labelrotation=20)
    ax.flatten()[0].set_ylabel('Population')
    ax.flatten()[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.flatten()[0].set_title('a) SIRDS simulation')
    ax.flatten()[0].legend()

    # Plot Rt
    sns.lineplot(x=dates, y=real_reproduction_number, label='Original data', legend=True, linestyle=line_styles[0],
                 ax=ax.flatten()[1])
    sns.lineplot(x=dates, y=reproduction_number_sird, label='Simulation', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[1])
    if max_date_to_fit is not None:
        ax.flatten()[1].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[1].axhline(1, 0, 1, linestyle='--', color='red')
    ax.flatten()[1].set_xlabel('Month/Year')
    ax.flatten()[1].xaxis.set_major_formatter(mask_date)
    ax.flatten()[1].tick_params(axis='x', labelrotation=20)
    ax.flatten()[1].set_ylabel('$R_{t}$')
    ax.flatten()[1].set_title('b) Effective reproduction number ($R_{t}$)')
    ax.flatten()[1].legend()

    # Plot new cases
    sns.lineplot(x=dates, y=real_new_cases, label='Original data (reported cases)', legend=True,
                 linestyle=line_styles[0], ax=ax.flatten()[2])
    sns.lineplot(x=dates[1:], y=I_new_cases, label='Simulation (infections)', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[2])
    if max_date_to_fit is not None:
        ax.flatten()[2].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[2].set_xlabel('Month/Year')
    ax.flatten()[2].xaxis.set_major_formatter(mask_date)
    ax.flatten()[2].tick_params(axis='x', labelrotation=20)
    ax.flatten()[2].set_ylabel('Events per 100,000 people')
    ax.flatten()[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.flatten()[2].set_title('c) New cases (infections)')
    ax.flatten()[2].legend()

    # Plot new deaths
    sns.lineplot(x=dates, y=real_new_deaths, label='Original data', legend=True, linestyle=line_styles[0],
                 ax=ax.flatten()[3])
    sns.lineplot(x=dates[1:], y=D_new_deaths, label='Simulation', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[3])
    if max_date_to_fit is not None:
        ax.flatten()[3].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[3].set_xlabel('Month/Year')
    ax.flatten()[3].xaxis.set_major_formatter(mask_date)
    ax.flatten()[3].tick_params(axis='x', labelrotation=20)
    ax.flatten()[3].set_ylabel('Deaths per 100,000 people')
    ax.flatten()[3].set_title('d) New deaths')
    ax.flatten()[3].legend()

    # Plot total cases
    sns.lineplot(x=dates, y=real_total_cases, label='Original data (reported cases)', legend=True,
                 linestyle=line_styles[0], ax=ax.flatten()[4])
    sns.lineplot(x=dates, y=I_accumulated, label='Simulation (infections)', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[4])
    if max_date_to_fit is not None:
        ax.flatten()[4].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[4].set_xlabel('Month/Year')
    ax.flatten()[4].xaxis.set_major_formatter(mask_date)
    ax.flatten()[4].tick_params(axis='x', labelrotation=20)
    ax.flatten()[4].set_ylabel('Events per 100,000 people')
    ax.flatten()[4].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.flatten()[4].set_title('e) Total cases (infections)')
    ax.flatten()[4].legend()

    # Plot total deaths
    sns.lineplot(x=dates, y=real_total_deaths, label='Original data', legend=True, linestyle=line_styles[0],
                 ax=ax.flatten()[5])
    sns.lineplot(x=dates, y=D, label='Simulation', legend=True, linestyle=line_styles[1], ax=ax.flatten()[5])
    if max_date_to_fit is not None:
        ax.flatten()[5].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[5].set_xlabel('Month/Year')
    ax.flatten()[5].xaxis.set_major_formatter(mask_date)
    ax.flatten()[5].tick_params(axis='x', labelrotation=20)
    ax.flatten()[5].set_ylabel('Deaths per 100,000 people')
    ax.flatten()[5].set_title('f) Total deaths')
    ax.flatten()[5].legend()

    fig.tight_layout()
    if save:
        plt.savefig(directory_to_save+'/sirds_result_'+id_in_file+'.pdf', bbox_inches="tight", transparent=True)
    plt.show()

    # Plot infecteds
    # t = np.linspace(0, len(I), len(I))
    # fig = plt.figure(facecolor='w')
    # ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    # ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    # ax.set_xlabel('Time /days')
    # ax.set_ylim(0, I.max())
    # ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(length=0)
    # legend = ax.legend()
    # legend.get_frame().set_alpha(0.5)
    # for spine in ('top', 'right', 'bottom', 'left'):
    #     ax.spines[spine].set_visible(False)
    # plt.show()

    # Plot fuzzy epidemic periods with fast transition
    # fig, ax = FuzzyVariableVisualizer(epidemic_periods_with_fast_transition_fuzzy_variable).view()
    # fig.set_size_inches(util.centimeter_to_inch(17.4), util.centimeter_to_inch(10.875))
    # ax.set_xlabel("Month/Year")
    # ax.legend(title="Epidemic period", bbox_to_anchor=(0.075, 1), ncol=5)
    # # ax.set_xticklabels(pd.DataFrame(dates[ax.get_xticks()[:-1].astype(int).tolist()])[0].dt.strftime('%m/%Y'))
    # fig.tight_layout()
    # if save:
    #     plt.savefig(directory_to_save+'/epidemic_periods_with_fast_transition_fuzzy_variable_'+id_in_file+'.pdf',
    #                 bbox_inches="tight", transparent=True)
    # plt.show()

    # Plot fuzzy epidemic periods with slow transition
    # fig, ax = FuzzyVariableVisualizer(epidemic_periods_with_slow_transition_fuzzy_variable).view()
    # fig.set_size_inches(util.centimeter_to_inch(17.4), util.centimeter_to_inch(10.875))
    # ax.set_xlabel("Month/Year")
    # ax.legend(title="Epidemic period", bbox_to_anchor=(0.075, 1), ncol=5)
    # # ax.set_xticklabels(pd.DataFrame(dates[ax.get_xticks()[:-1].astype(int).tolist()])[0].dt.strftime('%m/%Y'))
    # fig.tight_layout()
    # if save:
    #     plt.savefig(directory_to_save+'/epidemic_periods_with_slow_transition_fuzzy_variable_'+id_in_file+'.pdf', bbox_inches="tight",
    #                 transparent=True)
    # plt.show()

    # Plot fuzzy R0
    # estimated_days_between_infections = np.array([])
    # for i in range(min_length):
    #     estimation = epidemic_parameter_defuzzification(epidemic_periods_with_fast_transition_fuzzy_variable,
    #                                                     days_between_infections_values, i)
    #     estimated_days_between_infections = np.append(estimated_days_between_infections, [estimation])
    #
    # fig, ax = plt.subplots(figsize=(util.centimeter_to_inch(17.4), util.centimeter_to_inch(8.7)))
    # sns.lineplot(x=dates, y=days_to_recovery / estimated_days_between_infections, markers=False, color='black')
    # plt.ylabel("Fuzzy $R^{'}_{0}$")
    # ax.xaxis.set_major_formatter(mask_date)
    # plt.xlabel('Month/Year')
    # plt.xticks(rotation=20)
    # fig.tight_layout()
    # if save:
    #     plt.savefig(directory_to_save+'/fuzzy_r0_'+id_in_file+'.pdf', bbox_inches="tight", transparent=True)
    # plt.show()

    # Plot fuzzy IFR
    # estimated_case_fatality_probability = np.array([])
    # for i in range(min_length):
    #     estimation = epidemic_parameter_defuzzification(epidemic_periods_with_slow_transition_fuzzy_variable,
    #                                                     case_fatality_probability_values, i)
    #     estimated_case_fatality_probability = np.append(estimated_case_fatality_probability, [estimation])
    #
    # fig, ax = plt.subplots(figsize=(util.centimeter_to_inch(17.4), util.centimeter_to_inch(8.7)))
    # sns.lineplot(x=dates, y=estimated_case_fatality_probability * 100, markers=False, color='black')
    # plt.ylabel('Fuzzy IFR (%)')
    # ax.xaxis.set_major_formatter(mask_date)
    # plt.xlabel('Month/Year')
    # plt.xticks(rotation=20)
    # fig.tight_layout()
    # if save:
    #     plt.savefig(directory_to_save+'/fuzzy_ifr_'+id_in_file+'.pdf', bbox_inches="tight", transparent=True)
    # plt.show()

    # Plot fuzzy days to loss the immunity
    # estimated_loss_immunity_in_days = np.array([])
    # for i in range(min_length):
    #     estimation = epidemic_parameter_defuzzification(epidemic_periods_with_slow_transition_fuzzy_variable,
    #                                                     loss_immunity_in_days_values, i)
    #     estimated_loss_immunity_in_days = np.append(estimated_loss_immunity_in_days, [estimation])
    #
    # fig, ax = plt.subplots(figsize=(util.centimeter_to_inch(17.4), util.centimeter_to_inch(8.7)))
    # sns.lineplot(x=dates, y=estimated_loss_immunity_in_days, markers=False, color='black')
    # plt.ylabel('Fuzzy immunity duration\n(days)')
    # ax.xaxis.set_major_formatter(mask_date)
    # plt.xlabel('Month/Year')
    # plt.xticks(rotation=20)
    # fig.tight_layout()
    # if save:
    #     plt.savefig(directory_to_save+'/fuzzy_loss_immunity_'+id_in_file+'.pdf', bbox_inches="tight", transparent=True)
    # plt.show()

def plot_result(df_S, df_I, df_R, df_D, df_new_deaths, df_I_accumulated, real_new_deaths, real_total_deaths,
                real_reproduction_number, df_rt,
                real_total_cases, real_new_cases, df_new_cases, dates, directory_to_save='images',
                id_in_file='', max_date_to_fit=None):

    mask_date = mdates.DateFormatter('%m/%Y')
    line_styles = ['-', '--', ':', '-.', '-']
    plt.rc('font', size=8)
    sns.set_style("ticks")
    sns.set_palette(util.get_default_colors_categorical_seaborn())
    fig, ax = plt.subplots(3, 2, figsize=(util.centimeter_to_inch(19.05), util.centimeter_to_inch(14.288)), sharex=False)

    # Plot the data on three separate curves for S(t), I(t), R(t) and D(t)
    sns.lineplot(x=df_S['date'], y=df_S['S'], label='Susceptible', color=util.get_default_colors_categorical_seaborn()[1], legend=True,
                 linestyle=line_styles[0], ax=ax.flatten()[0], errorbar=('ci', 95))
    sns.lineplot(x=df_I['date'], y=df_I['I'], label='Infected', color=util.get_default_colors_categorical_seaborn()[2], legend=True,
                 linestyle=line_styles[1], ax=ax.flatten()[0], errorbar=('ci', 95))
    sns.lineplot(x=df_R['date'], y=df_R['R'], label='Recovered', color=util.get_default_colors_categorical_seaborn()[4], legend=True,
                 linestyle=line_styles[2], ax=ax.flatten()[0], errorbar=('ci', 95))
    sns.lineplot(x=df_D['date'], y=df_D['D'], label='Deceased', color=util.get_default_colors_categorical_seaborn()[3], legend=True,
                 linestyle=line_styles[3], ax=ax.flatten()[0], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[0].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[0].set_xlabel('Month/Year')
    ax.flatten()[0].xaxis.set_major_formatter(mask_date)
    ax.flatten()[0].tick_params(axis='x', labelrotation=20)
    ax.flatten()[0].set_ylabel('Population')
    ax.flatten()[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.flatten()[0].set_title('a) SIRDS simulation')
    ax.flatten()[0].legend()

    # Plot Rt
    sns.lineplot(x=dates, y=real_reproduction_number, label='Original data', legend=True, linestyle=line_styles[0],
                 ax=ax.flatten()[1])
    sns.lineplot(x=df_rt['date'], y=df_rt['rt'], label='Simulation', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[1], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[1].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[1].axhline(1, 0, 1, linestyle='--', color='red')
    ax.flatten()[1].set_xlabel('Month/Year')
    ax.flatten()[1].xaxis.set_major_formatter(mask_date)
    ax.flatten()[1].tick_params(axis='x', labelrotation=20)
    ax.flatten()[1].set_ylabel('$R_{t}$')
    ax.flatten()[1].set_title('b) Effective reproduction number ($R_{t}$)')
    ax.flatten()[1].legend()

    # Plot new cases
    sns.lineplot(x=dates, y=real_new_cases, label='Original data (reported cases)', legend=True,
                 linestyle=line_styles[0], ax=ax.flatten()[2])
    sns.lineplot(x=df_new_cases['date'], y=df_new_cases['cases'], label='Simulation (infections)', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[2], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[2].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[2].set_xlabel('Month/Year')
    ax.flatten()[2].xaxis.set_major_formatter(mask_date)
    ax.flatten()[2].tick_params(axis='x', labelrotation=20)
    ax.flatten()[2].set_ylabel('Events per 100,000 people')
    ax.flatten()[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.flatten()[2].set_title('c) New cases (infections)')
    ax.flatten()[2].legend()

    # Plot new deaths
    sns.lineplot(x=dates, y=real_new_deaths, label='Original data', legend=True, linestyle=line_styles[0],
                 ax=ax.flatten()[3])
    sns.lineplot(x=df_new_deaths['date'], y=df_new_deaths['deaths'], label='Simulation', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[3], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[3].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[3].set_xlabel('Month/Year')
    ax.flatten()[3].xaxis.set_major_formatter(mask_date)
    ax.flatten()[3].tick_params(axis='x', labelrotation=20)
    ax.flatten()[3].set_ylabel('Deaths per 100,000 people')
    ax.flatten()[3].set_title('d) New deaths')
    ax.flatten()[3].legend()

    # Plot total cases
    sns.lineplot(x=dates, y=real_total_cases, label='Original data (reported cases)', legend=True,
                 linestyle=line_styles[0], ax=ax.flatten()[4])
    sns.lineplot(x=df_I_accumulated['date'], y=df_I_accumulated['I_accumulated'], label='Simulation (infections)', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[4], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[4].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[4].set_xlabel('Month/Year')
    ax.flatten()[4].xaxis.set_major_formatter(mask_date)
    ax.flatten()[4].tick_params(axis='x', labelrotation=20)
    ax.flatten()[4].set_ylabel('Events per 100,000 people')
    ax.flatten()[4].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.flatten()[4].set_title('e) Total cases (infections)')
    ax.flatten()[4].legend()

    # Plot total deaths
    sns.lineplot(x=dates, y=real_total_deaths, label='Original data', legend=True, linestyle=line_styles[0],
                 ax=ax.flatten()[5])
    sns.lineplot(x=df_D['date'], y=df_D['D'], label='Simulation', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[5], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[5].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[5].set_xlabel('Month/Year')
    ax.flatten()[5].xaxis.set_major_formatter(mask_date)
    ax.flatten()[5].tick_params(axis='x', labelrotation=20)
    ax.flatten()[5].set_ylabel('Deaths per 100,000 people')
    ax.flatten()[5].set_title('f) Total deaths')
    ax.flatten()[5].legend()

    fig.tight_layout()
    filename = 'images/result_output'+id_in_file
    plt.savefig(filename+'.pdf', bbox_inches="tight")
    plt.savefig(filename+'.tiff', format='tiff', dpi=300, transparent=False, bbox_inches='tight')
    img = Image.open(filename+".tiff")
    img.save(filename+"_compressed.tiff", compression="tiff_lzw")
    plt.show()

def plot_result_simple(df_S, df_I, df_R, df_D, df_new_deaths, df_I_accumulated, real_new_deaths, real_total_deaths,
                real_reproduction_number, df_rt,
                real_total_cases, real_new_cases, df_new_cases, dates, directory_to_save='images',
                id_in_file='', max_date_to_fit=None):

    # Set the desired dates for major ticks on the x-axis
    desired_dates = ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']
    mask_date = mdates.DateFormatter('%m/%Y')
    line_styles = ['-', '--', ':', '-.', '-']
    plt.rc('font', size=8)
    sns.set_style("ticks")
    sns.set_palette(util.get_default_colors_categorical_seaborn())
    fig, ax = plt.subplots(1, 3, figsize=(util.centimeter_to_inch(19.05), util.centimeter_to_inch(4.42)), sharex=False)

    # Plot the data on three separate curves for S(t), I(t), R(t) and D(t)
    sns.lineplot(x=df_S['date'], y=df_S['S'], label='S', color=util.get_default_colors_categorical_seaborn()[1], legend=True,
                 linestyle=line_styles[0], ax=ax.flatten()[0], errorbar=('ci', 95))
    sns.lineplot(x=df_I['date'], y=df_I['I'], label='I', color=util.get_default_colors_categorical_seaborn()[2], legend=True,
                 linestyle=line_styles[1], ax=ax.flatten()[0], errorbar=('ci', 95))
    sns.lineplot(x=df_R['date'], y=df_R['R'], label='R', color=util.get_default_colors_categorical_seaborn()[4], legend=True,
                 linestyle=line_styles[2], ax=ax.flatten()[0], errorbar=('ci', 95))
    sns.lineplot(x=df_D['date'], y=df_D['D'], label='D', color=util.get_default_colors_categorical_seaborn()[3], legend=True,
                 linestyle=line_styles[3], ax=ax.flatten()[0], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[0].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[0].set_xlabel('Date')
    ax.flatten()[0].set_xticks(desired_dates)
    ax.flatten()[0].xaxis.set_major_formatter(mask_date)
    ax.flatten()[0].tick_params(axis='x', labelrotation=0)
    ax.flatten()[0].set_ylabel('Population')
    ax.flatten()[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.flatten()[0].set_title('a) SIRDS simulation')

    # Plot Rt
    sns.lineplot(x=dates, y=real_reproduction_number, label='Original data', legend=True, linestyle=line_styles[0],
                 ax=ax.flatten()[1])
    sns.lineplot(x=df_rt['date'], y=df_rt['rt'], label='Simulation', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[1], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[1].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[1].axhline(1, 0, 1, linestyle='--', color='red')
    ax.flatten()[1].set_xlabel('Date')
    ax.flatten()[1].set_xticks(desired_dates)
    ax.flatten()[1].xaxis.set_major_formatter(mask_date)
    ax.flatten()[1].tick_params(axis='x', labelrotation=0)
    ax.flatten()[1].set_ylabel('$R_{t}$')
    ax.flatten()[1].set_title('b) Effective reproduction number ($R_{t}$)')

    # Plot new deaths
    sns.lineplot(x=dates, y=real_new_deaths, label='Original data', legend=True, linestyle=line_styles[0],
                 ax=ax.flatten()[2])
    sns.lineplot(x=df_new_deaths['date'], y=df_new_deaths['deaths'], label='Simulation', legend=True, linestyle=line_styles[1],
                 ax=ax.flatten()[2], errorbar=('ci', 95))
    if max_date_to_fit is not None:
        ax.flatten()[2].axvline(max_date_to_fit, 0, 1, linestyle=':', color='gray')
    ax.flatten()[2].set_xlabel('Date')
    ax.flatten()[2].set_xticks(desired_dates)
    ax.flatten()[2].xaxis.set_major_formatter(mask_date)
    ax.flatten()[2].tick_params(axis='x', labelrotation=0)
    ax.flatten()[2].set_ylabel('Deaths\n(per 100K people)')
    ax.flatten()[2].set_title('c) New deaths')

    # Creating custom legend entries and labels
    legend_a = mlines.Line2D([], [], color='black', linestyle='', label='Chart a:')
    legend_b = mlines.Line2D([], [], color='black', linestyle='', label='Chart b:')
    legend_c = mlines.Line2D([], [], color='black', linestyle='', label='Chart c:')
    legend_empty = mlines.Line2D([], [], color='black', linestyle='', label='')

    # Getting the handles and labels from the plot
    handles_0, labels_0 = ax.flatten()[0].get_legend_handles_labels()
    handles_1, labels_1 = ax.flatten()[1].get_legend_handles_labels()
    handles_2, labels_2 = ax.flatten()[2].get_legend_handles_labels()

    handles = ([legend_a, legend_b, legend_c] +
               [handles_0[0]] + [handles_1[0]] + [handles_2[0]] +
               [handles_0[1]] + [handles_1[1]] + [handles_2[1]] +
               [handles_0[2]] + [legend_empty, legend_empty] +
               [handles_0[3]] + [legend_empty, legend_empty])
    labels = (['Chart a:', 'Chart b:', 'Chart c:'] +
               [labels_0[0]] + [labels_1[0]] + [labels_2[0]] +
               [labels_0[1]] + [labels_1[1]] + [labels_2[1]] +
               [labels_0[2]] + ['', ''] +
               [labels_0[3]] + ['', ''])

    # Combining the legend elements with the existing handles and labels
    fig.legend(handles=handles, labels=labels,loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.025), title='Legend')
    ax.flatten()[0].legend([], frameon=False)
    ax.flatten()[1].legend([], frameon=False)
    ax.flatten()[2].legend([], frameon=False)

    fig.tight_layout()
    filename = 'images/result_output_simple'+id_in_file
    plt.savefig(filename+'.pdf', bbox_inches="tight")
    plt.savefig(filename+'.tiff', format='tiff', dpi=300, transparent=False, bbox_inches='tight')
    img = Image.open(filename+".tiff")
    img.save(filename+"_compressed.tiff", compression="tiff_lzw")
    plt.show()

def _calculate_performance(real_new_deaths, D_new_deaths, real_reproduction_number, reproduction_number_sird):
    if len(real_reproduction_number[~np.isnan(real_reproduction_number)]) > 0:
        mae = get_error_deaths_rt(D_new_deaths,
                            real_new_deaths,
                            reproduction_number_sird,
                            real_reproduction_number)

        indices_to_remove = np.argwhere(np.isnan(real_reproduction_number))
        real_reproduction_number = np.delete(real_reproduction_number, indices_to_remove)
        reproduction_number_sird_train = np.delete(reproduction_number_sird, indices_to_remove)
        sse_Rt = mean_squared_error(real_reproduction_number, reproduction_number_sird_train)
        r2_Rt = r2_score(real_reproduction_number, reproduction_number_sird_train)
    else:
        mae = None
        sse_Rt = None
        r2_Rt = None

    sse_D = mean_squared_error(D_new_deaths, real_new_deaths)
    r2_D = r2_score(D_new_deaths, real_new_deaths)

    return mae, sse_D, r2_D, sse_Rt, r2_Rt

def calculate_performance(real_new_deaths, D_new_deaths, real_reproduction_number, reproduction_number_sird, train_period=None):
    if train_period is None:
        train_period = len(real_new_deaths)

    if len(D_new_deaths) < len(real_new_deaths):
        D_fitted = D_new_deaths[:train_period-1]
        real_new_deaths_fitted = real_new_deaths[1:train_period]
    else:
        D_fitted = D_new_deaths[1:train_period]
        real_new_deaths_fitted = real_new_deaths[1:train_period]

    reproduction_number_sird_fitted = reproduction_number_sird[:train_period]
    real_reproduction_number_fitted = real_reproduction_number[:train_period]
    mae_fit, sse_D_fit, r2_D_fit, sse_Rt_fit, r2_Rt_fit = _calculate_performance(
        real_new_deaths_fitted, D_fitted, real_reproduction_number_fitted, reproduction_number_sird_fitted)

    if train_period < len(real_new_deaths):
        if len(D_new_deaths) < len(real_new_deaths):
            D_predicted = D_new_deaths[train_period-1:]
        else:
            D_predicted = D_new_deaths[train_period:]
        real_new_deaths_predicted = real_new_deaths[train_period:]
        D_predicted = D_predicted[:len(real_new_deaths_predicted)]

        real_reproduction_number_predicted = real_reproduction_number[train_period:]
        reproduction_number_sird_predicted = reproduction_number_sird[train_period:]
        reproduction_number_sird_predicted = reproduction_number_sird_predicted[:len(real_reproduction_number_predicted)]

        mae_predicton, sse_D_predicton, r2_D_predicton, sse_Rt_predicton, r2_Rt_predicton = _calculate_performance(
            real_new_deaths_predicted, D_predicted, real_reproduction_number_predicted, reproduction_number_sird_predicted)

        real_new_deaths_predicted_month_1 = real_new_deaths_predicted[:30]
        D_predicted_month_1 = D_predicted[:30]
        real_reproduction_number_predicted_month_1 = real_reproduction_number_predicted[:30]
        reproduction_number_sird_predicted_month_1 = reproduction_number_sird_predicted[:30]
        mae_predicton_month_1, sse_D_predicton_month_1, r2_D_predicton_month_1, sse_Rt_predicton_month_1, r2_Rt_predicton_month_1 = _calculate_performance(
            real_new_deaths_predicted_month_1, D_predicted_month_1, real_reproduction_number_predicted_month_1, reproduction_number_sird_predicted_month_1)

        real_new_deaths_predicted_month_2 = real_new_deaths_predicted[30:60]
        D_predicted_month_2 = D_predicted[30:60]
        real_reproduction_number_predicted_month_2 = real_reproduction_number_predicted[30:60]
        reproduction_number_sird_predicted_month_2 = reproduction_number_sird_predicted[30:60]
        mae_predicton_month_2, sse_D_predicton_month_2, r2_D_predicton_month_2, sse_Rt_predicton_month_2, r2_Rt_predicton_month_2 = _calculate_performance(
            real_new_deaths_predicted_month_2, D_predicted_month_2, real_reproduction_number_predicted_month_2, reproduction_number_sird_predicted_month_2)

        real_new_deaths_predicted_month_3 = real_new_deaths_predicted[60:]
        D_predicted_month_3 = D_predicted[60:]
        real_reproduction_number_predicted_month_3 = real_reproduction_number_predicted[60:]
        reproduction_number_sird_predicted_month_3 = reproduction_number_sird_predicted[60:]
        mae_predicton_month_3, sse_D_predicton_month_3, r2_D_predicton_month_3, sse_Rt_predicton_month_3, r2_Rt_predicton_month_3 = (
                _calculate_performance(real_new_deaths_predicted_month_3, D_predicted_month_3, real_reproduction_number_predicted_month_3, reproduction_number_sird_predicted_month_3))

        return mae_fit, sse_D_fit, r2_D_fit, sse_Rt_fit, r2_Rt_fit, mae_predicton, sse_D_predicton, r2_D_predicton, sse_Rt_predicton, r2_Rt_predicton, mae_predicton_month_1, sse_D_predicton_month_1, r2_D_predicton_month_1, sse_Rt_predicton_month_1, r2_Rt_predicton_month_1, mae_predicton_month_2, sse_D_predicton_month_2, r2_D_predicton_month_2, sse_Rt_predicton_month_2, r2_Rt_predicton_month_2, mae_predicton_month_3, sse_D_predicton_month_3, r2_D_predicton_month_3, sse_Rt_predicton_month_3, r2_Rt_predicton_month_3
    else:
        return mae_fit, sse_D_fit, r2_D_fit, sse_Rt_fit, r2_Rt_fit

def plot_parameters(df_r0, df_IFR, df_days_to_loss_immunity, country):
    mask_date = mdates.DateFormatter('%m/%Y')
    plt.rc('font', size=6)
    style = dict(color='black')
    sns.set_style("ticks")

    fig, ax = plt.subplots(1, 3, figsize=(util.centimeter_to_inch(19.05), util.centimeter_to_inch(4.22)))

    # Plot fuzzy R0
    sns.lineplot(x=df_r0['date'], y=df_r0['r0'], markers=False, color='black', errorbar=('ci', 95), ax=ax[0])
    ax[0].set_ylabel("$R_{0}(t)$")
    ax[0].xaxis.set_major_formatter(mask_date)
    ax[0].set_xlabel('Month/Year')
    ax[0].tick_params(axis='x', rotation=45)
    ax[0].set_title('a) Time-varying basic reproduction number $R_{0}(t)$')

    # Plot fuzzy IFR
    sns.lineplot(x=df_IFR['date'], y=df_IFR['ifr']*100, markers=False, color='black', errorbar=('ci', 95), ax=ax[1])
    ax[1].set_ylabel("IFR(t) (in %)")
    ax[1].xaxis.set_major_formatter(mask_date)
    ax[1].set_xlabel('Month/Year')
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].set_title('b) Time-varying Infection Fatality Rate IFR(t)')

    # Plot fuzzy days to loss the immunity
    sns.lineplot(x=df_days_to_loss_immunity['date'], y=df_days_to_loss_immunity['Omega'], markers=False, color='black',
                 errorbar=('ci', 95), ax=ax[2])
    ax[2].set_ylabel("$\Omega(t)$ (in days)")
    ax[2].xaxis.set_major_formatter(mask_date)
    ax[2].set_xlabel('Month/Year')
    ax[2].tick_params(axis='x', rotation=45)
    ax[2].set_title('c) Time-varying days to loss of immunity $\Omega(t)$')

    fig.tight_layout()

    filename = 'images/result_parameters_'+country
    plt.savefig(filename+'.pdf', bbox_inches="tight")
    plt.savefig(filename+'.tiff', format='tiff', dpi=300, transparent=False, bbox_inches='tight')
    img = Image.open(filename+".tiff")
    img.save(filename+"_compressed.tiff", compression="tiff_lzw")

    plt.show()