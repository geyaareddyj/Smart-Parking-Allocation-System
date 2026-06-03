import streamlit as st
import heapq
import random
import pandas as pd
from collections import deque

# ==============================================================================
# GLOBAL AGENT SPECIFICATION MATRIX: SYSTEM PEAS PROFILE
# ==============================================================================
# P (Performance): Maximize total allocation utility, minimize path retrieval metric
#                  costs g(n), eliminate constraint violations, minimize state sync
#                  delays, and maximize real-time robustness under exogenous anomalies.
# E (Environment): Discrete, static topology grid with dynamically updating slot nodes;
#                  Stochastic/Non-deterministic transitions caused by rogue drivers;
#                  Partially Observable status under un-telemetered manual actions.
# A (Actuators):   Mutex data state lock commands, real-time trajectory array outputs,
#                  database synchronization re-routing operations.
# S (Sensors):     Gate telemetry arrays (vehicle weight/scale profile, ID signatures,
#                  charging station resource requirements), database slot state arrays.
# ==============================================================================

# Set up browser page layouts
st.set_page_config(
    page_title="AI Smart Parking System",
    page_icon="🚗",
    layout="wide"
)

# ==============================================================================
# MODULE 1 - AUTOMATED ENVIRONMENT STATE SPACE & GRAPH TOPOLOGY
# PEAS Alignment -> Environment (E): Graph Model Definition & Spatial Matrix Layout
# ==============================================================================
class PhysicalParkingGraph:
    """
    Defines the spatial state-space configuration of the parking environment.
    Transforms structural roadways and intersection connections into a formal 
    topological adjacency list with accurate coordinate geometry mapping.
    """
    def __init__(self):
        # Physical coordinates utilized for computing the admissible geometric heuristic h(n)
        self.coordinates = {
            "Entrance": (0, 0), "Main Lane": (2, 0),
            "Zone A Lane": (2, 2), "Zone B Lane": (4, 2),
            "A1": (1, 3), "A2": (2, 3), "B1": (4, 3), "B2": (5, 3)
        }
        self.graph = {}

    def add_roadway(self, src, dest, distance):
        """Constructs an undirected, weighted transition edge between spatial nodes."""
        if src not in self.graph: self.graph[src] = {}
        if dest not in self.graph: self.graph[dest] = {}
        self.graph[src][dest] = distance
        self.graph[dest][src] = distance  # Bi-directional routing channels


# ==============================================================================
# MODULE 2 - ADMISSIBLE TRAJECTORY OPTIMIZATION & INFORMED PATHFINDING
# PEAS Alignment -> Actuators (A): Computes optimal path outputs to target nodes
# ==============================================================================
class NavigationSearchEngine:
    """
    Core algorithmic movement subsystem. Evaluates physical paths using Uninformed 
    (BFS), Cost-Optimal (UCS), and Heuristically Optimized (A*) search frameworks.
    """
    def __init__(self, parking_graph):
        self.graph = parking_graph.graph
        self.coords = parking_graph.coordinates

    def manhattan_heuristic(self, current, goal):
        """Mathematical Heuristic h(n): Computes absolute Manhattan Distance layout."""
        x1, y1 = self.coords[current]
        x2, y2 = self.coords[goal]
        return abs(x1 - x2) + abs(y1 - y2)

    def a_star_search(self, start, goal):
        """
        Informed Search Strategy: Optimizes trajectory using global evaluation metrics
        f(n) = g(n) + h(n). Guaranteed optimal as h(n) remains strictly admissible.
        """
        priority_queue = [(0 + self.manhattan_heuristic(start, goal), 0, start, [start])]
        visited = set()

        while priority_queue:
            _, g_cost, node, path = heapq.heappop(priority_queue)
            if node == goal: return g_cost, path
            if node not in visited:
                visited.add(node)
                for neighbor, weight in self.graph.get(node, {}).items():
                    g = g_cost + weight
                    h = self.manhattan_heuristic(neighbor, goal)
                    f = g + h
                    heapq.heappush(priority_queue, (f, g, neighbor, path + [neighbor]))
        return float("inf"), []


# ==============================================================================
# MODULE 3 - CONSTRAINT SATISFACTION PROBLEM (CSP) STRUCTURAL ENGINE
# PEAS Alignment -> Performance Measure (P): Elimination of hard constraint violations
# ==============================================================================
class ParkingConstraintEngine:
    """
    Enforces absolute compliance across physical system constraints (Hard Constraints).
    Performs algorithmic domain pruning to remove non-compliant variable states.
    """
    def __init__(self, database):
        self.db = database

    def evaluate_csp(self, vehicle_size, needs_ev, display_container):
        """Prunes the available state variables based on strict unary and binary criteria."""
        valid_spots = {}
        display_container.markdown("### 🛑 Module 3: Constraint Satisfaction Domain Pruning")
        
        for spot_id, attr in self.db.items():
            if attr["occupied"]:
                display_container.write(f"❌ *Pruned State {spot_id}*: Fails Occupancy Status Rule.")
                continue
            if vehicle_size.lower() == "suv" and attr["size"].lower() == "compact":
                display_container.write(f"❌ *Pruned State {spot_id}*: Fails Dimension Constraint (SUV Scale mismatch).")
                continue
            if needs_ev and not attr["has_ev"]:
                display_container.write(f"❌ *Pruned State {spot_id}*: Fails Hardware Constraint (EV Charger unavailable).")
                continue
            
            display_container.write(f"✅ *Allowed State {spot_id}*: Passes all active constraint checks.")
            valid_spots[spot_id] = attr
        return valid_spots


# ==============================================================================
# MODULE 4 - MULTI-ATTRIBUTE UTILITY DECISION-MAKING MATRIX
# PEAS Alignment -> Performance Measure (P): Optimizing user preference comfort scores
# ==============================================================================
class PreferenceUtilityEngine:
    """
    Resolves soft-constraint variables using multi-attribute utility theory calculations.
    Rank-orders candidate nodes based on preference weights.
    """
    @staticmethod
    def calculate_utility(valid_candidates, display_container):
        """Formulates global utility calculation: U(s) = Proximity_Value + Comfort_Value."""
        display_container.markdown("### 📈 Module 4: Multi-Criteria Utility Matrix")
        utility_scores = {}
        
        for spot_id, attr in valid_candidates.items():
            proximity_utility = (40 - attr["distance"])
            comfort_utility = 15 if attr["has_shade"] else 0
            total_utility = proximity_utility + comfort_utility
            utility_scores[spot_id] = total_utility
            display_container.write(f"🔹 *Node {spot_id} Metrics* ➔ Proximity: {proximity_utility} pts | Structural Shade: {comfort_utility} pts | **Global Utility U(s) = {total_utility}**")
            
        return utility_scores


# ==============================================================================
# MODULE 5 - STOCHASTIC REASONING & PROBABILISTIC UNCERTAINTY FILTER
# PEAS Alignment -> Environment (E): Models non-deterministic transitions
# ==============================================================================
class StateUncertaintyEngine:
    """
    Computes system risk levels and implements conditional probabilities to model
    stochastic actions within partially observable environments.
    """
    def __init__(self, database):
        self.db = database

    def calculate_lot_congestion_probability(self, display_container):
        """Computes current structural traffic distribution: P(Delay | Occupancy Ratio)."""
        total_spots = len(self.db)
        occupied_spots = sum(1 for spot in self.db.values() if spot["occupied"])
        p_delay = round(occupied_spots / total_spots, 2)
        
        display_container.markdown("### 🎲 Module 5: Probabilistic Cognition Under Uncertainty")
        display_container.write(f"📊 Physical Density State: `{occupied_spots}/{total_spots}` occupied allocations.")
        display_container.write(f"⚠️ Calculated Bayesian Risk Factor: `P(Congestion Delay) = {p_delay}`")
        return p_delay


# ==============================================================================
# MODULE 6 - CENTRAL INTEGRATED STREAMLIT PIPELINE CONTROLLER
# PEAS Alignment -> Sensors (S) Input Capture & Actuators (A) Processing Coordination
# ==============================================================================

st.title("🚗 AI Smart Parking Allocation Dashboard")
st.markdown("A production-grade, multi-module intelligent framework leveraging CSP filtering, Multi-Attribute Utility theory, and $f(n) = g(n) + h(n)$ A* trajectory calculations.")
st.write("---")

# Persistent State Management for the Parking Database
if "parking_lot_db" not in st.session_state:
    st.session_state.parking_lot_db = {
        "A1": {"size": "compact", "has_ev": False, "distance": 10, "has_shade": True,  "occupied": False},
        "A2": {"size": "suv",     "has_ev": False, "distance": 15, "has_shade": False, "occupied": False},
        "B1": {"size": "compact", "has_ev": True,  "distance": 5,  "has_shade": True,  "occupied": True}, # Unavailable node
        "B2": {"size": "suv",     "has_ev": True,  "distance": 20, "has_shade": True,  "occupied": False}
    }

# Initialization of Environmental Graph Topology
map_graph = PhysicalParkingGraph()
map_graph.add_roadway("Entrance", "Main Lane", 5)
map_graph.add_roadway("Main Lane", "Zone A Lane", 4)
map_graph.add_roadway("Main Lane", "Zone B Lane", 8)
map_graph.add_roadway("Zone A Lane", "A1", 2)
map_graph.add_roadway("Zone A Lane", "A2", 3)
map_graph.add_roadway("Zone B Lane", "B1", 2)
map_graph.add_roadway("Zone B Lane", "B2", 4)

# Web Browser Column Layout Split Configurations
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🏁 Gate Operations Panel (Sensors S)")
    vehicle_id = st.text_input("Vehicle Identifier Signature", "SUV-01")
    size = st.selectbox("Vehicle Scale Profile", ["Compact", "SUV"])
    needs_ev = st.checkbox("Requires Electric Vehicle (EV) Charging Hardware")
    
    allocate_btn = st.button("Run Smart Allocation Pipeline", type="primary")
    
    if st.button("Reset Parking Lot Map States"):
        for spot in st.session_state.parking_lot_db:
            st.session_state.parking_lot_db[spot]["occupied"] = False
        st.session_state.parking_lot_db["B1"]["occupied"] = True
        st.rerun()

    st.write("---")
    st.subheader("🗺️ Module 1: Live Environmental State Monitor")
    df = pd.DataFrame(st.session_state.parking_lot_db).T
    st.dataframe(df, use_container_width=True)

with col2:
    if allocate_btn:
        st.subheader("🛰️ Real-Time Multi-Module Agent Trace")
        
        # Display layout components
        csp_box = st.container(border=True)
        utility_box = st.container(border=True)
        search_box = st.container(border=True)
        uncertainty_box = st.container(border=True)
        
        # 1. Run CSP Filter (Module 3)
        csp_engine = ParkingConstraintEngine(st.session_state.parking_lot_db)
        eligible_spots = csp_engine.evaluate_csp(size, needs_ev, csp_box)
        
        if not eligible_spots:
            csp_box.error("❌ **Pipeline Aborted**: Zero open slots match architectural requirements.")
        else:
            # 2. Run Utility Score Calculation (Module 4)
            utility_matrix = PreferenceUtilityEngine.calculate_utility(eligible_spots, utility_box)
            optimal_target_spot = max(utility_matrix, key=utility_matrix.get)
            utility_box.success(f"🎯 **Optimal Selection**: Spot `{optimal_target_spot}` chosen with Global Utility U(s) = `{utility_matrix[optimal_target_spot]}`.")
            
            # Lock the spot state via Mutex Simulation (Actuators A)
            st.session_state.parking_lot_db[optimal_target_spot]["occupied"] = True
            
            # 3. Generate Path via A* Search (Module 2)
            navigator = NavigationSearchEngine(map_graph)
            g_cost, navigation_route = navigator.a_star_search("Entrance", optimal_target_spot)
            
            search_box.markdown("### 🛰️ Module 2: Informed Navigation Optimization")
            search_box.info(f"🛣️ **Calculated Optimal Trajectory Array (A*)**:  \n`{' ➔ '.join(navigation_route)}`")
            search_box.metric(label="Calculated Trajectory Metric Cost g(n)", value=f"{g_cost} meters")
            
            # 4. Check for Environmental Uncertainty (Module 5)
            uncertainty_engine = StateUncertaintyEngine(st.session_state.parking_lot_db)
            p_delay = uncertainty_engine.calculate_lot_congestion_probability(uncertainty_box)
            
            if p_delay >= 0.75:
                uncertainty_box.error(f"🔴 System alert! High Traffic Delay Risk index detected.")
            elif p_delay >= 0.5:
                uncertainty_box.warning(f"🟡 Moderate path routing bottlenecks predicted.")
            else:
                uncertainty_box.success(f"🟢 Path clearance values optimal. Low delay thresholds.")
                
            # Exogenous Behavior Anomaly Detection (Stochastic Transition Analysis)
            random.seed(len(vehicle_id))
            if random.random() < 0.15 and len(eligible_spots) > 1:
                other_options = [s for s in eligible_spots if s != optimal_target_spot]
                stolen_spot = random.choice(other_options)
                
                # Undo lock and update systemic variables to represent reality
                st.session_state.parking_lot_db[optimal_target_spot]["occupied"] = False
                st.session_state.parking_lot_db[stolen_spot]["occupied"] = True
                
                uncertainty_box.error(f"⚠️ **Exogenous Behavior Anomaly Detected!** \nDriver deviated and occupied node `{stolen_spot}` instead of target `{optimal_target_spot}`. Partially observable model parameters resynchronized.")
            else:
                st.balloons()
                uncertainty_box.success(f"🏁 **Execution Terminated**: Vehicle successfully synchronized inside spot `{optimal_target_spot}`.")
    else:
        st.info("💡 Fill out the Gate Operations Form on the left side and click the button to see the AI agent run its real-time multi-module calculations.")
