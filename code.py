import random
import heapq

# ==============================================================================
# THE "PEAS" SYSTEM
# P (Performance): Find the absolute best spot and the shortest path.
# E (Environment): The parking lot lanes and available parking spots.
# A (Actuators): The system locking a spot and printing out the path.
# S (Sensors): The gate entrance where a driver inputs their car details.
# ==============================================================================

# ── MODULE 1: THE PARKING LOT DATA (THE ENVIRONMENT) ──
# We represent the parking lot as a simple Python dictionary.
# Each spot has specific rules: Size, EV charger availability, Distance, and Shade.
PARKING_LOT_DATABASE = {
    "A1": {"size": "compact", "has_ev": False, "distance": 10, "has_shade": True,  "occupied": False},
    "A2": {"size": "suv",     "has_ev": False, "distance": 15, "has_shade": False, "occupied": False},
    "B1": {"size": "compact", "has_ev": True,  "distance": 5,  "has_shade": True,  "occupied": True},  # Already full
    "B2": {"size": "suv",     "has_ev": True,  "distance": 20, "has_shade": True,  "occupied": False}
}

# The actual physical grid layout (Coordinates) used by our A* pathfinder to calculate distance
NODE_COORDINATES = {
    "Entrance": (0, 0), "Main Lane": (2, 0),
    "Zone A": (2, 2), "Zone B": (4, 2),
    "A1": (1, 3), "A2": (2, 3), "B1": (4, 3), "B2": (5, 3)
}


# ── MODULE 2: TRAJECTORY OPTIMIZATION (A* PATHFINDING) ──
def calculate_manhattan_distance(current_node, target_node):
    """
    HEURISTIC h(n): This is a smart guess of the distance using a grid map.
    It calculates the 'Manhattan Distance' (absolute walk path across a city block).
    """
    x1, y1 = NODE_COORDINATES[current_node]
    x2, y2 = NODE_COORDINATES[target_node]
    return abs(x1 - x2) + abs(y1 - y2)


def run_a_star_navigation(start, goal):
    """
    REAL A* SEARCH

    f(n) = g(n) + h(n)

    g(n): Actual cost from the start node
    h(n): Manhattan distance heuristic
    """

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}

    # Distance from start to each node
    g_score = {node: float('inf') for node in GRAPH}
    g_score[start] = 0

    # Estimated total cost
    f_score = {node: float('inf') for node in GRAPH}
    f_score[start] = calculate_manhattan_distance(start, goal)

    while open_set:

        _, current = heapq.heappop(open_set)

        # Goal reached
        if current == goal:

            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            path.reverse()

            return g_score[goal], path

        # Explore neighbouring nodes
        for neighbor in GRAPH[current]:

            # Assume each road segment costs 1 unit
            tentative_g = g_score[current] + 1

            if tentative_g < g_score[neighbor]:

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f_score[neighbor] = (
                    tentative_g
                    + calculate_manhattan_distance(neighbor, goal)
                )

                heapq.heappush(
                    open_set,
                    (f_score[neighbor], neighbor)
                )

    return None, []


# ── MODULE 3: CONSTRAINT SATISFACTION PROBLEM (CSP) ──
def run_csp_filter(vehicle_size, requires_ev):
    """
    HARD CONSTRAINT ENFORCEMENT: Strips away spots that are completely illegal.
    """
    print("\n🛑 [MODULE 3] Running CSP Filter (Checking Hard Constraints)...")
    valid_candidates = {}

    for spot_id, features in PARKING_LOT_DATABASE.items():
        # Rule 1: If a spot is already taken, kick it out!
        if features["occupied"]:
            print(f"  ❌ Spot {spot_id} removed: Already occupied.")
            continue
        
        # Rule 2: If the car is too big (SUV) and the spot is tiny (Compact), kick it out!
        if vehicle_size == "suv" and features["size"] == "compact":
            print(f"  ❌ Spot {spot_id} removed: Too small for an SUV.")
            continue
            
        # Rule 3: If the car needs charging, but the spot has no charger, kick it out!
        if requires_ev and not features["has_ev"]:
            print(f"  ❌ Spot {spot_id} removed: Lacks electric vehicle hardware.")
            continue

        # If it passes all checks, it's a valid option!
        print(f"  ✅ Spot {spot_id} is VALID and added to allowed options.")
        valid_candidates[spot_id] = features

    return valid_candidates


# ── MODULE 4: MULTI-ATTRIBUTE UTILITY THEORY (MAUT) ──
def run_utility_scoring(valid_spots):
    """
    SOFT PREFERENCE CALCULATOR: Out of the legal spots, which one is 'the best' choice?
    Formula: Utility Score = Proximity (Closeness) + Comfort (Shade)
    """
    print("\n📈 [MODULE 4] Evaluating Spot Utility Scores...")
    best_spot = None
    highest_score = -999

    for spot_id, features in valid_spots.items():
        # Closeness points: Closer spots (smaller distance) get higher points
        proximity_points = 40 - features["distance"]
        
        # Comfort points: Give 15 extra bonus points if there is a shade tree!
        comfort_points = 15 if features["has_shade"] else 0
        
        # Total overall score
        total_utility = proximity_points + comfort_points
        print(f"  🔹 Spot {spot_id} Utility = {proximity_points} (Proximity) + {comfort_points} (Shade) ➔ Total: {total_utility} pts")

        if total_utility > highest_score:
            highest_score = total_utility
            best_spot = spot_id

    return best_spot


# ── MODULE 5: UNCERTAINTY & RISK ANALYSIS ──
def check_system_uncertainty():
    """
    PROBABILISTIC COGNITION: Calculates how crowded the lot is right now.
    Probability of Delay = Occupied Spots / Total Spots
    """
    print("\n🎲 [MODULE 5] Analyzing Environmental Risk Under Uncertainty...")
    total_spots = len(PARKING_LOT_DATABASE)
    occupied_count = sum(1 for spot in PARKING_LOT_DATABASE.values() if spot["occupied"])
    
    probability_of_delay = occupied_count / total_spots
    print(f"  📊 Lot Congestion: {occupied_count}/{total_spots} spots filled.")
    print(f"  ⚠️ P(Traffic Delay Risk) = {probability_of_delay:.2f}")


# ── MODULE 6: MAIN SYSTEM EXECUTIVE PIPELINE CONTROL ──
def main():
    print("==================================================")
    print("      AI SMART PARKING ALLOCATION SIMULATOR       ")
    print("==================================================")
    
    # STEP 1: Capture user inputs (Simulating our System Sensors)
    car_id = input("Enter Vehicle Plate/ID: ")
    car_size = input("Enter Car Size (compact/suv): ").lower().strip()
    ev_check = input("Does it need EV Charging? (yes/no): ").lower().strip()
    needs_ev = True if ev_check == "yes" else False

    # STEP 2: Filter options by enforcing rules (CSP Engine)
    allowed_spots = run_csp_filter(car_size, needs_ev)

    if not allowed_spots:
        print("\n❌ PIPELINE CRASHED: No available spots match your car's physical requirements!")
        return

    # STEP 3: Select the highest performing choice (Utility Decision Matrix)
    chosen_spot = run_utility_scoring(allowed_spots)
    print(f"\n🎯 SYSTEM DECISION: Spot '{chosen_spot}' has been chosen as the absolute optimal spot!")

    # Lock the spot state (Simulating our System Actuators updates)
    PARKING_LOT_DATABASE[chosen_spot]["occupied"] = True

    # STEP 4: Navigate to the spot using math routing paths (A* Search Optimization)
    travel_cost, turn_by_turn_route = run_a_star_navigation("Entrance", chosen_spot)
    print(f"\n🛣️ [MODULE 2] A* Search Path Calculated: {' ➔ '.join(turn_by_turn_route)}")
    print(f"  📏 Optimal Travel Metric Cost: {travel_cost} meters")

    # STEP 5: Run a quick statistical diagnostic of the system metrics
    check_system_uncertainty()
    
    print("\n🏁 EXECUTION COMPLETED: Vehicle safely assigned and logged.")
    print("=========================================================")

if __name__ == "__main__":
    main()