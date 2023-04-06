from utility import get_rates, get_trial_solution, nspp
import SimFunctions
import SimRNG
import SimClasses
import numpy as np


RENTBIKE = "Rent a bike"
RETURNBIKE = "Return a bike"
REBALANCE = "rebalancing"
FINISH = "EndSimulation"
SINGLE_TRIP_FEE = 3.75  # CAD per bike per ride
LOSS = 3.75  # Canadian Dollar for cost of loss of business opportunity
FUEL_COST = 3  # Canadian Dollar for rebalancing vehicles carrying bikes
THRESHOLD = 2.0  # schedule the "rebalancing" once the number of bikes is less than the threshold
BIKES_IN_STATION_AFTER_REBALANCING = 5.0  # the number of bikes in the station would reach 5 after rebalancing
TRANSPORT_DELAY = 20  # it takes 20 minutes for bikes to be transported to the station for bike rebalancing
NUMBER_OF_SIMULATION = 1  # number of simulation runs
OPERATION_COST = 2  # Canadian Dollar per bike for operation
RUN_LENGTH = 180  # 3 hours

MaxRentingRate, RentingRate, MaxReturnRate, ReturnRate = get_rates()

trial_solution = get_trial_solution()


def rental(**kwargs):
    global revenue, loss
    # determine which station is being simulated
    station_id = kwargs['stationID']
    pickup_queue = kwargs["pickup_queue"]
    return_queue = kwargs["return_queue"]
    # Use NSPP to schedule the next bike rental event for current station
    SimFunctions.schedule(Calendar, RENTBIKE,
                          nspp(station_id, MaxRentingRate, RentingRate),
                          stationID=station_id, pickup_queue=TheQueues[station_id],
                          return_queue=TheQueues[station_id+5])

    # Checks if there are people waiting to put the bikes back.
    if return_queue.num_queue() > 0:
        departing_customer = return_queue.remove()
        waiting_time = SimClasses.Clock - departing_customer.CreateTime
        Wait.record(waiting_time)
        # If customer has waited too long to return the bike, refund is given.
        # The customer rents the bike and the people waiting to put the bikes back returns a bike.
        if waiting_time > 5:
            loss += SINGLE_TRIP_FEE

    # If no one is waiting to put the bikes back,
    # check if there are bikes available and update number of bikes for the station.
    else:
        if num_bikes[station_id] > 0:
            num_bikes[station_id] -= 1
            # Customer pays to rent bike.
            revenue += SINGLE_TRIP_FEE
        # No bikes available so the customer begins waiting for bikes to become available.
        else:
            pickup_queue.add(SimClasses.Entity())  # add new customer

    # Schedule the bike rebalancing event once the number of bikes is less than the threshold
    """
    There are two conditions to be met for the rebalancing operation. Schedule the rebalancing operation 
    only if the number of bikes is less than the threshold and the previous rebalance operation has ended. 
    """
    if num_bikes[station_id] <= THRESHOLD and quantity[station_id] == 0:
        # quantity : how much does it need to transport/rebalance?
        quantity[station_id] = BIKES_IN_STATION_AFTER_REBALANCING - num_bikes[station_id]
        SimFunctions.schedule(Calendar, REBALANCE, TRANSPORT_DELAY, stationID=station_id,
                              pickup_queue=TheQueues[station_id], return_queue=TheQueues[station_id+5],
                              num_ordered=quantity[station_id], signal=-1)


def ride_end(**kwargs):
    global revenue, loss
    station_id = kwargs['stationID']
    return_queue = kwargs['return_queue']
    pickup_queue = kwargs['pickup_queue']
    # Use NSPP to schedule the next end of ride for current station
    SimFunctions.schedule(Calendar, RETURNBIKE, nspp(station_id, MaxReturnRate, ReturnRate),
                          stationID=station_id, pickup_queue=TheQueues[station_id],
                          return_queue=TheQueues[station_id + 5])

    # We assume not every customer will wait for a bike and eventually take a bike.
    # Customers leave after 5 min.
    while True:
        # Check if customers are waiting.
        if pickup_queue.num_queue() == 0:
            # If no one is waiting to pick up the bike,check if there are empty racks to keep the bike.
            if num_bikes[station_id] < Num_docks[station_id]:
                # Customer returns the bike to the rack.
                num_bikes[station_id] += 1
            # No empty racks. The customer begins waiting in queue.
            else:
                return_queue.add(SimClasses.Entity())
            break
        # If there are customers waiting to pick up the bike,
        # check if the customer has left due to waiting too long to pick up the bike.
        departing_customer = pickup_queue.remove()
        waiting_time = SimClasses.Clock - departing_customer.CreateTime
        Wait.record(SimClasses.Clock - departing_customer.CreateTime)

        if waiting_time < 5:
            # Next waiting customer gets a bike from the customer who has finished the ride.
            # Break the while loop.
            break
        else:
            # We lose a customer because the customer has waited too long
            loss += LOSS

    if return_queue.num_queue() >= THRESHOLD and quantity[station_id] == 0:
        quantity[station_id] = return_queue.num_queue()
        SimFunctions.schedule(Calendar, REBALANCE, TRANSPORT_DELAY, stationID=station_id,
                              pickup_queue=TheQueues[station_id], return_queue=TheQueues[station_id + 5],
                              num_ordered=quantity[station_id], signal=1)


def rebalance(**kwargs):
    global revenue, cost, loss
    station_id = kwargs['stationID']
    return_queue = kwargs['return_queue']
    pickup_queue = kwargs['pickup_queue']
    # The number of bikes needed for bike rebalancing
    num_ordered = kwargs['num_ordered']
    signal = kwargs['signal']
    # Cost due to bike rebalancing
    cost += (num_ordered * OPERATION_COST) + FUEL_COST
    # Signal is -1 means the station is almost out of bikes for customers renting bikes,
    # then the center will carry bikes to the station
    if signal == -1:
        # The number of bikes changes after the rebalancing operation
        num_bikes[station_id] += num_ordered
        # We assume not every customer will wait for a bike and eventually take a bike. Customers leave after 5 min.
        while True:
            # Check whether customers are waiting in the pick_up queue and whether
            # the ordered number of bikes is in short supply.
            if (pickup_queue.num_queue() == 0.0) | (num_bikes[station_id] == 0.0):
                break

            departing_customer = pickup_queue.remove()
            waiting_time = SimClasses.Clock - departing_customer.CreateTime
            Wait.record(SimClasses.Clock - departing_customer.CreateTime)
            if waiting_time < 5:
                # Next waiting customer gets a bike
                num_bikes[station_id] -= 1
            else:
                # We lose a customer
                loss += LOSS
    # Signal is +1 means the station is out of racks for customers returning bikes,
    # then the center will carry bikes back to the center
    else:
        while True:
            # Check if customers are waiting in the Return queue
            if (return_queue.num_queue() == 0.0) | (num_ordered == 0.0):
                break

            departing_customer = return_queue.remove()
            num_ordered -= 1
            waiting_time = SimClasses.Clock - departing_customer.CreateTime
            Wait.record(SimClasses.Clock - departing_customer.CreateTime)
            if waiting_time > 5:
                # If customer has waited too long to return the bike, refund is given.
                loss += SINGLE_TRIP_FEE
        num_bikes[station_id] -= num_ordered
    # Set the num_ordered to zero so the current rebalancing operation is over.
    quantity[station_id] = 0


if __name__ == '__main__':
    Calendar = SimClasses.EventCalendar()
    Wait = SimClasses.DTStat()

    TheCTStats = []
    TheDTStats = []
    TheQueues = []
    TheResources = []

    # 5 for pickup queue, 5 for return queue
    for i in range(10):
        TheQueues.append(SimClasses.FIFOQueue())

    TheDTStats.append(Wait)

    AllWaitMean = np.zeros((NUMBER_OF_SIMULATION, len(trial_solution)))

    ZSimRNG = SimRNG.InitializeRNSeed()

    # Create a 2d array to record the balance of each trial solution in each simulation run
    output = np.zeros((NUMBER_OF_SIMULATION, len(trial_solution)))
    for k in range(NUMBER_OF_SIMULATION):
        for j in range(len(trial_solution)):
            num_bikes = []
            initial_Num_bikes = []
            # Apply to our model to each trial solution
            for i in range(0, 5, 1):
                num_bikes.append(trial_solution[j][i])
                initial_Num_bikes.append(trial_solution[j][i])
            # Number of bike racks (total) at each station. This is the maximum parking
            # capacity for every station.
            # Num_docks = [15, 27, 11, 15, 15]
            Num_docks = [30, 54, 22, 30, 30]
            # initial number of ordered bikes for rebalancing
            quantity = [0, 0, 0, 0, 0]
            cost = 0
            loss = 0
            revenue = 0

            SimClasses.Clock = 0.0
            SimFunctions.sim_funcs_init(Calendar, TheQueues, TheCTStats, TheDTStats, TheResources)
            for i in range(0, 5, 1):
                # Use NSPP to schedule the first arrival for each station.
                # Initialize queues at each station.
                SimFunctions.schedule(Calendar, RENTBIKE,
                                      nspp(i, MaxRentingRate, RentingRate),
                                      stationID=i, pickup_queue=TheQueues[i], return_queue=TheQueues[i + 5])
                SimFunctions.schedule(Calendar, RETURNBIKE,
                                      nspp(i, MaxReturnRate, ReturnRate),
                                      stationID=i, pickup_queue=TheQueues[i], return_queue=TheQueues[i + 5])

            SimFunctions.schedule(Calendar, FINISH, RUN_LENGTH)
            NextEvent = Calendar.Remove()
            # time is updating with next event time
            SimClasses.Clock = NextEvent.EventTime

            if NextEvent.EventType == RENTBIKE:
                rental(**NextEvent.kwargs)
            elif NextEvent.EventType == RETURNBIKE:
                ride_end(**NextEvent.kwargs)
            elif NextEvent.EventType == REBALANCE:
                rebalance(**NextEvent.kwargs)

            while NextEvent.EventType != FINISH:
                NextEvent = Calendar.Remove()
                SimClasses.Clock = NextEvent.EventTime
                if NextEvent.EventType == RENTBIKE:
                    rental(**NextEvent.kwargs)
                elif NextEvent.EventType == RETURNBIKE:
                    ride_end(**NextEvent.kwargs)
                elif NextEvent.EventType == REBALANCE:
                    rebalance(**NextEvent.kwargs)

            # cost for repositioning bike overnight
            num_repositioning_bikes = (abs(np.array(num_bikes) - np.array(initial_Num_bikes)).sum())
            cost += (num_repositioning_bikes * OPERATION_COST) + FUEL_COST
            output[k][j] = float(revenue - cost - loss)
            AllWaitMean[k][j] = Wait.mean()

    print(output)

    # Get the expectation by calculating the mean over all simulation runs for each trial solution
    Estimated_Expected_wait_time = np.mean(AllWaitMean, axis=0)
    Estimated_Expected_balance = np.mean(output, axis=0)

    # get the maximum profit subject to a waiting time constraint
    # aim to max balance and min waiting time
    i = Estimated_Expected_balance[0] / Estimated_Expected_wait_time[0]
    for x, y in zip(Estimated_Expected_balance, Estimated_Expected_wait_time):
        # control waiting time within 5 min
        if ((x/y) >= i) & (y <= 5):
            i = x
    best = np.argwhere(Estimated_Expected_balance == i)
    print(best)
