import numpy as np
from django.shortcuts import render
from matplotlib import pyplot as plt
from scipy.optimize import linprog
import io
import base64
import matplotlib
import logging
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

matplotlib.use('Agg')  # Use 'Agg' backend for matplotlib

# Configure logging
logger = logging.getLogger(__name__)

def home(request):
    return render(request, 'home.html')

def graphical_method(request):
    return render(request, 'graphical_method.html')

def graphical_steps(request):
    return render(request, 'graphical_steps.html')

def graphical_application(request):
    return render(request, 'graphical_application.html')

def graphical_solve(request):
    if request.method == 'POST':
        try:
            # Get optimization type
            opt_type = request.POST.get('opt_type', 'maximize')
            c = [float(x) for x in request.POST.get('objective', '').split()]
            if len(c) != 2:
                raise ValueError("Graphical method supports only two variables.")

            constraints = request.POST.getlist('constraints[]')
            A_ub, b_ub = [], []

            for constraint in constraints:
                parts = constraint.strip().split()
                if len(parts) != 3:
                    raise ValueError("Each constraint must have exactly two coefficients and a RHS value.")
                A_ub.append([float(parts[0]), float(parts[1])])
                b_ub.append(float(parts[2]))

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)

            # For maximization, convert to minimization by multiplying c by -1
            if opt_type == 'maximize':
                c_lp = [-ci for ci in c]
            else:
                c_lp = c

            # Set bounds for variables (non-negative variables)
            bounds = [(0, None), (0, None)]  # Bounds for x₁ and x₂

            # Solve LP using linprog
            res = linprog(c=c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

            context = {}

            # Prepare for plotting
            fig, ax = plt.subplots()
            x1_max = max(b_ub) * 1.1 if len(b_ub) > 0 else 10
            x1_vals = np.linspace(0, x1_max, 400)

            # Plot the constraints
            for i in range(len(A_ub)):
                a1, a2 = A_ub[i]
                b_i = b_ub[i]
                if a2 != 0:
                    x2_vals = (b_i - a1 * x1_vals) / a2
                    x2_vals = np.maximum(0, x2_vals)  # Ensure x₂ >= 0
                    constraint_label = f'{a1}x₁ + {a2}x₂ ≤ {b_i}'
                    ax.plot(x1_vals, x2_vals, label=constraint_label)
                else:
                    x = b_i / a1
                    constraint_label = f'{a1}x₁ ≤ {b_i}'
                    ax.axvline(x=x, label=constraint_label)

            # Shade feasible region
            x1_grid = np.linspace(0, x1_max, 500)
            x2_grid = np.linspace(0, x1_max, 500)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            Z = np.ones_like(X1, dtype=bool)

            for i in range(len(A_ub)):
                a1, a2 = A_ub[i]
                Z &= (a1 * X1 + a2 * X2 <= b_ub[i])

            Z &= (X1 >= 0)
            Z &= (X2 >= 0)

            # Check if feasible region exists
            if not np.any(Z):
                ax.text(0.5, 0.5, 'No Feasible Region', fontsize=15, ha='center', va='center', transform=ax.transAxes)
                context['no_solution'] = True
                context['solution_message'] = "Not possible: No feasible region exists with the given constraints."
            else:
                ax.contourf(X1, X2, Z, levels=[0.5, 1], colors=['#a0ffa0'], alpha=0.3)

                if res.success:
                    x1_opt, x2_opt = res.x
                    if opt_type == 'maximize':
                        objective_value = -res.fun  # Multiply by -1 because we minimized
                    else:
                        objective_value = res.fun

                    # Round the values
                    x1_opt_rounded = round(x1_opt, 4)
                    x2_opt_rounded = round(x2_opt, 4)
                    objective_value_rounded = round(objective_value, 4)

                    # Log the optimal solution for debugging
                    logger.debug(f"Optimal Solution: x1={x1_opt}, x2={x2_opt}, objective_value={objective_value}")
                    print(f"Optimal Solution: x1={x1_opt}, x2={x2_opt}, objective_value={objective_value}")

                    # Plot optimal solution point
                    ax.plot(x1_opt, x2_opt, 'ro', label='Optimal Solution')
                    annotation_text = f'Optimal Solution\nx₁ = {x1_opt_rounded}\nx₂ = {x2_opt_rounded}\nZ = {objective_value_rounded}'
                    ax.annotate(annotation_text, xy=(x1_opt, x2_opt),
                                xytext=(x1_opt + 0.5, x2_opt + 0.5),
                                arrowprops=dict(facecolor='black', shrink=0.05))

                    # Prepare context with solution values
                    context['x1_opt'] = x1_opt_rounded
                    context['x2_opt'] = x2_opt_rounded
                    context['objective_value'] = objective_value_rounded
                    context['no_solution'] = False
                else:
                    # No optimal solution found within feasible region
                    ax.text(0.5, 0.5, 'No Optimal Solution', fontsize=15, ha='center', va='center', transform=ax.transAxes)
                    context['no_solution'] = True
                    context['solution_message'] = "Not possible: An optimal solution does not exist for the given constraints."

            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title('Graphical Solution')
            ax.legend()
            ax.grid(True)
            ax.set_xlim(0, x1_max)
            ax.set_ylim(0, x1_max)

            # Save the plot to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            plt.close(fig)
            buf.seek(0)
            graph = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            # Add the graph to the context
            context['graph'] = graph
            context['error'] = None  # No errors occurred

            # Render the result template with the context
            return render(request, 'graphical_result.html', context)

        except Exception as e:
            # Log the exception
            logger.error(f"Exception in graphical_solve view: {e}", exc_info=True)
            print(f"Exception occurred: {e}")
            error_message = f"An error occurred: {str(e)}. Please ensure your inputs are correct and try again."
            return render(request, 'graphical_solve.html', {'error': error_message})

    return render(request, 'graphical_solve.html')
# Load Simplex Method Page
import numpy as np
from django.shortcuts import render

def simplex_method(request):
    return render(request, 'simplex_method.html')

def simplex_steps(request):
    return render(request, 'simplex_steps.html')

def simplex_solve(request):
    return render(request, 'simplex_solve.html')

def simplex_application(request):
    return render(request, 'simplex_application.html')

def simplex(c, A, b, maximize=True):
    """
    Implements the Simplex Method for Linear Programming.
    :param c: Coefficients of the objective function.
    :param A: Constraint coefficients matrix.
    :param b: Right-hand side constants of constraints.
    :param maximize: Boolean flag for maximizing (True) or minimizing (False).
    :return: Optimal solution dictionary and optimal value.
    """
    num_constraints, num_variables = A.shape
    slack_vars = np.eye(num_constraints)

    tableau = np.hstack((A, slack_vars, b.reshape(-1, 1)))
    obj_row = np.hstack(((-1 if maximize else 1) * c, np.zeros(num_constraints + 1)))
    tableau = np.vstack((tableau, obj_row))

    while True:
        if np.all(tableau[-1, :-1] >= 0):
            break  # Optimal solution found

        pivot_col = np.argmin(tableau[-1, :-1])  # Entering variable
        column = tableau[:-1, pivot_col]

        # Avoid division by zero: only compute ratios where column > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(column > 0, tableau[:-1, -1] / column, np.inf)

        if np.all(np.isinf(ratios)):
            raise ValueError("The problem is unbounded.")

        pivot_row = np.argmin(ratios)
        pivot_element = tableau[pivot_row, pivot_col]

        if pivot_element == 0:
            raise ValueError("Pivot element is zero; numerical instability encountered.")

        # Normalize pivot row
        tableau[pivot_row, :] /= pivot_element

        # Eliminate pivot column from other rows
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                multiplier = tableau[i, pivot_col]
                tableau[i, :] -= multiplier * tableau[pivot_row, :]

    # Extract solution
    solution = np.zeros(num_variables)
    for i in range(num_constraints):
        col = tableau[i, :num_variables]
        if np.count_nonzero(col) == 1:
            var_index = np.where(col == 1)[0][0]
            solution[var_index] = tableau[i, -1]

    optimal_value = tableau[-1, -1]
    return {f"x{i+1}": round(solution[i], 6) for i in range(num_variables)}, round(optimal_value, 6)

def simplex_solver(request):
    if request.method == 'POST':
        try:
            maximize = request.POST.get('type') == 'maximize'
            num_variables = int(request.POST.get('variables'))
            c = np.array([float(x) for x in request.POST.get('objective').split()])

            constraints = request.POST.getlist('constraints[]')
            A, b = [], []

            for constraint in constraints:
                parts = constraint.split()
                if len(parts) < num_variables + 1:
                    raise ValueError("Each constraint must have enough coefficients.")

                A.append([float(x) for x in parts[:-1]])
                b.append(float(parts[-1]))

            A = np.array(A)
            b = np.array(b)

            solution, optimal_value = simplex(c, A, b, maximize)

            return render(request, 'simplex_result.html', {
                'solution': solution,
                'optimal_value': optimal_value
            })

        except ValueError as e:
            return render(request, 'simplex_result.html', {'error': str(e)})

    return render(request, 'simplex_solve.html')


# Transportation Problem Views
def transportation_method(request):
    return render(request, 'transportation_method.html')

def transportation_steps(request):
    return render(request, 'transportation_steps.html')

def transportation_application(request):
    return render(request, 'transportation_application.html')

def transportation_solve(request):
    if request.method == 'POST':
        try:
            method = request.POST.get('method')
            supply = [int(x) for x in request.POST.get('supply').split()]
            demand = [int(x) for x in request.POST.get('demand').split()]
            cost_matrix = []
            for i in range(len(supply)):
                row = [int(x) for x in request.POST.get(f'row_{i}').split()]
                if len(row) != len(demand):
                    raise ValueError("Each cost row must have the same number of elements as demand nodes.")
                cost_matrix.append(row)
            cost_matrix = np.array(cost_matrix)

            if sum(supply) != sum(demand):
                raise ValueError("Total supply must equal total demand.")

            if method == 'north_west':
                result = north_west_corner_method(supply.copy(), demand.copy(), cost_matrix)
            elif method == 'least_cost':
                result = least_cost_method(supply.copy(), demand.copy(), cost_matrix)
            elif method == 'vogels':
                result = vogels_approximation_method(supply.copy(), demand.copy(), cost_matrix)
            else:
                raise ValueError("Invalid method selected.")

            return render(request, 'transportation_result.html', {'result': result})
        except Exception as e:
            return render(request, 'transportation_result.html', {'error': str(e)})
    else:
        return render(request, 'transportation_solve.html')

def north_west_corner_method(supply, demand, cost):
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))
    i = j = 0
    while i < m and j < n:
        alloc = min(supply[i], demand[j])
        allocation[i][j] = alloc
        supply[i] -= alloc
        demand[j] -= alloc
        if supply[i] == 0:
            i += 1
        elif demand[j] == 0:
            j += 1
    total_cost = np.sum(allocation * cost)
    return {'allocation': allocation, 'total_cost': total_cost}

def least_cost_method(supply, demand, cost):
    m, n = len(supply), len(demand)
    allocation = np.zeros((m, n))
    cost_copy = cost.copy()
    while np.any(supply) and np.any(demand):
        min_cost = np.min(cost_copy[cost_copy != np.inf])
        indices = np.where(cost_copy == min_cost)
        i, j = indices[0][0], indices[1][0]
        alloc = min(supply[i], demand[j])
        allocation[i][j] = alloc
        supply[i] -= alloc
        demand[j] -= alloc
        cost_copy[i][j] = np.inf
        if supply[i] == 0:
            cost_copy[i, :] = np.inf
        if demand[j] == 0:
            cost_copy[:, j] = np.inf
    total_cost = np.sum(allocation * cost)
    return {'allocation': allocation, 'total_cost': total_cost}

def vogels_approximation_method(supply, demand, cost):
    return "Vogel's Approximation Method is not yet implemented."

#Space Mission

def terraform_home(request):
    habitability = None
    atmosphere = temperature = gravity = None

    if request.method == "POST":
        atmosphere = request.POST.get("atmosphere")
        temperature = float(request.POST.get("temperature"))
        gravity = float(request.POST.get("gravity"))

        # Simple logic to compute habitability (adjust as needed)
        score = 100

        # Atmosphere scoring
        if atmosphere == "oxygen-rich":
            score -= 0
        elif atmosphere == "carbon-dioxide":
            score -= 30
        elif atmosphere == "methane":
            score -= 50
        elif atmosphere == "none":
            score -= 70

        # Temperature scoring (ideal is 15°C to 25°C)
        if temperature < -50 or temperature > 60:
            score -= 40
        elif temperature < 0 or temperature > 35:
            score -= 20

        # Gravity scoring (ideal is 9.8 m/s² ± 2)
        if gravity < 5 or gravity > 15:
            score -= 20

        habitability = max(score, 0)

    return render(request, 'terraform_planet.html', {
        'habitability': habitability,
        'atmosphere': atmosphere,
        'temperature': temperature,
        'gravity': gravity,
    })



def mission_dashboard(request):
    destinations = ["Moon", "Mars", "Europa", "Titan"]
    if request.method == "POST":
        destination = request.POST["destination"]
        fuel = float(request.POST["fuel"])
        cargo = float(request.POST["cargo"])
        
        # Dummy success probability based on fuel/cargo balance
        success_probability = min(1.0, (fuel / (cargo + 1)) * 0.8)
        
        return render(request, 'mission_dashboard.html', {
            "destination": destination,
            "fuel": fuel,
            "cargo": cargo,
            "success_probability": round(success_probability * 100, 2),
        })

    return render(request, "mission_dashboard.html", {"destinations": destinations})
