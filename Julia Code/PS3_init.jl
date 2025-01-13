# Defining parameters
beta = 0.9
r = 0.2
Ny = 50        # Number of Grid Points
Na = 100        # Number of Grid Points
b = 1


################ INITIALIZATION ##############################################################################

##Defining the Grid for the Endogenous State Variable: Capital
ymin = 5; ymax = 10;    # Bounds for Grid
grid_y = collect(range(ymin, ymax, length = Ny)); # Grid for income
amin = -5; amax = 5;    # Bounds for Grid
grid_a = collect(range(amin, amax, length = Na)); # Grid for assets

# No credit constraint 
v_opt = zeros(Ny,Ny);
index_a_opt= Array{Int64,2}(undef,Ny,Ny); # Hint! 
c_1_opt = zeros(Ny,Ny);
c_2_opt = zeros(Ny,Ny);
a_opt = zeros(Ny,Ny);

# With credit constraint
v_opt_bis = zeros(Ny,Ny);
c_1_opt_bis = zeros(Ny,Ny);
c_2_opt_bis = zeros(Ny,Ny);
a_opt_bis = zeros(Ny,Ny);

################ NO CREDIT CONSTRAINT ##############################################################################

# Fill the V array for all possible choice of a 
V = zeros(Ny,Ny,Na);
for ia in 1:Na
    for iy1 in 1:Ny
        for iy2 in 1:Ny
            # compute the consumption given y1, y2, a
            c_1 =  grid_y[iy1] - grid_a[ia]
            c_2 =  grid_y[iy2] + (1+r)*grid_a[ia]
            if c_1 > 0 && c_2 > 0 ## check if the consumptions are negative
                V[iy1, iy2, ia] = log(c_1) + beta * log(c_2)
            else
                V[iy1, iy2, ia] = -10^9  
            end
        end
    end
end

# Max and recover a_opt
for iy1 in 1:Ny
    for iy2 in 1:Ny
        max_V, index_a_opt[iy1, iy2] = findmax(V[iy1, iy2, :])
        
        # Store the maximum utility and corresponding optimal asset choice
        v_opt[iy1, iy2] = max_V
        
        # Retrieve the optimal asset and compute c1, c2
        optimal_a = grid_a[index_a_opt[iy1, iy2]]
        a_opt[iy1, iy2] = optimal_a
        c_1_opt[iy1, iy2] = grid_y[iy1] - optimal_a
        c_2_opt[iy1, iy2] = grid_y[iy2] + (1 + r) * optimal_a
    end
end

################ WITH CREDIT CONSTRAINT ##############################################################################

a_opt_bis = copy(a_opt)

binding_indices = findall(x -> x < -b, a_opt_bis)

for I in binding_indices
    # Convert CartesianIndex to tuple to access individual indices
    iy1, iy2 = Tuple(I)
    a_opt_bis[iy1, iy2] = -b
end
for iy1 in 1:Ny
    for iy2 in 1:Ny
        # Retrieve income levels for this iteration
        y1 = grid_y[iy1]
        y2 = grid_y[iy2]
        
        # Get the (possibly constrained) asset choice
        a = a_opt_bis[iy1, iy2]
        
        # Calculate consumption levels under the credit constraint
        c_1_opt_bis[iy1, iy2] = y1 - a
        c_2_opt_bis[iy1, iy2] = y2 + (1 + r) * a
        
        # Check if consumption levels are feasible
        if c_1_opt_bis[iy1, iy2] > 0 && c_2_opt_bis[iy1, iy2] > 0
            # Calculate utility if consumption levels are feasible
            v_opt_bis[iy1, iy2] = log(c_1_opt_bis[iy1, iy2]) + beta * log(c_2_opt_bis[iy1, iy2])
        else
            # Assign a large negative utility for infeasible choices
            v_opt_bis[iy1, iy2] = -1e9
        end
    end
end



################ PLOTS ##############################################################################
using Plots


plot(grid_y,a_opt[:,40],title="Value Function", label="No credit constraint")
plot!(grid_y,a_opt_bis[:,40],title="Value Function", label="With credit constraint")


plot(grid_y,v_opt[:,40],title="Asset Policy Function", label="No credit constraint")
plot!(grid_y,v_opt_bis[:,40],title="Asset Policy Function", label="With credit constraint")

