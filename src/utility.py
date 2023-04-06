import pandas as pd
import SimRNG
import SimClasses


# def get_rates():
#     df = pd.read_csv('predicted_data.csv')
#     df.index = df.start_hour
#     df = df.drop('start_hour', axis=1)
#
#     max_renting = []
#     rentings = []
#     max_return = []
#     returns = []
#
#     for i in range(0, 5):
#         max_renting.append(df['hourly_count'].max())
#         rentings.append(df.hourly_count.tolist())
#         # XXX degisilecek??????
#         max_return.append(df['hourly_count'].max())
#         returns.append(df.hourly_count.tolist())
#
#     return max_renting, rentings, max_return, returns

def get_rates():
    max_renting = []
    rentings = []
    max_return = []
    returns = []

    for i in range(0, 5):
        df = pd.read_csv('../data/rent_predicted_data_{ind}.csv'.format(ind=i))
        df.index = df.start_hour
        df = df.drop('start_hour', axis=1)
        max_renting.append(df['hourly_count'].max())
        rentings.append(df.hourly_count.tolist())

    for i in range(0, 5):
        df = pd.read_csv('../data/return_predicted_data_{ind}.csv'.format(ind=i))
        df.index = df.stop_hour
        df = df.drop('stop_hour', axis=1)
        max_return.append(df['hourly_count'].max())
        returns.append(df.hourly_count.tolist())
    return max_renting, rentings, max_return, returns


def get_trial_solution():
    trial_solution = []
    for a in range(10, 15):
        for b in range(10, 27):
            for c in range(10, 11):
                for d in range(10, 15):
                    for e in range(10, 15):
                        # Initial number of bikes available in the system.
                        if (a + b + c + d + e) == 70:
                            trial_solution.append([a, b, c, d, e])
    return trial_solution


# Prevent index out of range
# def pw_arr_rate(station_id, possible_arrival, renting_rates):
#     hour = int(possible_arrival / 20)
#     if hour <= 8:
#         return renting_rates[station_id][hour]
#     else:
#         return renting_rates[station_id][-1]


def nspp(station_id, max_renting_rates, renting_rates):
    possible_arrival = SimClasses.Clock + SimRNG.Expon(1 / (max_renting_rates[station_id] / 20), 1)
    # while SimRNG.Uniform(0, 1, 1) >= pw_arr_rate(station_id, possible_arrival, renting_rates)/(max_renting_rates[station_id]):
    while SimRNG.Uniform(0, 1, 1) >= \
            renting_rates[station_id][int(possible_arrival / 20)] / (max_renting_rates[station_id]):
        possible_arrival = possible_arrival + SimRNG.Expon(1 / (max_renting_rates[station_id] / 20), 1)
    return possible_arrival - SimClasses.Clock