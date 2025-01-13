##############################################################################################################################
## BLOCK 0 ##############################################################################################################################
##############################################################################################################################
using Distributions, LinearAlgebra

##Define Parameters
struct par_model
    beta::Float64              # discount factor
    mu::Float64            # risk aversion from CRRA parameter
    alpha::Float64            # capital share
    delta::Float64            # depreciation
    A::Float64            # aggregate productivity
    maxits::Int64
    tol::Float64
end

par = par_model(0.95, 2.0, 0.3, 0.03, 1.0, 3000, 1e-6)

##Defining the Grid for the Endogenous State Variable: Capital
Na = 300;
b = -0.2;
amax = 60;
agrid = collect(range(b, length=Na, stop=amax));

##Defining the Grid for the Exogenous State Variable: Technology Shock
rho = 0.9             # persistence of the AR(1) process
sigma = 0.2              # standard deviation of the AR(1) process
Ns = 5
function tauchen(mean, sd, rho, num_states; q=3)

    uncond_sd = sd / sqrt(1 - rho^2)
    y = range(-q * uncond_sd, stop=q * uncond_sd, length=num_states)
    d = y[2] - y[1]

    Pi = zeros(num_states, num_states)

    for row = 1:num_states
        # end points
        Pi[row, 1] = cdf(Normal(), (y[1] - rho * y[row] + d / 2) / sd)
        Pi[row, num_states] = 1 - cdf(Normal(), (y[num_states] - rho * y[row] - d / 2) / sd)

        # middle columns
        for col = 2:num_states-1
            Pi[row, col] = (cdf(Normal(), (y[col] - rho * y[row] + d / 2) / sd) -
                            cdf(Normal(), (y[col] - rho * y[row] - d / 2) / sd))
        end
    end

    yy = y .+ mean # center process around its mean

    Pi = Pi ./ sum(Pi, dims=2) # renormalize

    return Pi, yy
end

prob = tauchen(0, sigma, rho, Ns)[1];
logs = tauchen(0, sigma, rho, Ns)[2];
sgrid = exp.(logs);
Vguess = zeros(Ns, Na)

##############################################################################################################################
## BLOCK 1 ##############################################################################################################################
##############################################################################################################################

VFI = function (r, w, agrid, sgrid, V0, prob, par)

    Ns = length(sgrid)
    Na = length(agrid)

    U = zeros(Ns, Na, Na)

    for is in 1:Ns                     # Loop Over skills Today
        for ia in 1:Na                 # Loop Over assets Today
            for ia_p in 1:Na           # Loop Over assets Tomorrow
                s = sgrid[is]         # Technology Today
                a = agrid[ia]         # Capital Today
                a_p = agrid[ia_p]     # Capital Tomorrow
                # Solve for Consumption at Each Point
                c = (1 + r) * a + s * w - a_p
                if c .< 0
                    U[is, ia, ia_p] = -10^6
                else
                    ()
                    U[is, ia, ia_p] = c^(1 - par.mu) / (1 - par.mu)
                end
            end
        end
    end

    ### TO FILL ####### 
    # VFI loop
    # Initial Guess of the Value Function
    V0 = zeros(Ns, Na) # Initialize the value function to zero
    V0_bis = zeros(Ns, Na) # Initialize another value function to zero

    # Calculate the Guess of the Expected Value Function
    EVf = prob[:, :]' * V0_bis # Calculate the expected value function using the transition matrix and initial guess

    tol = 1e-4 
    maxits = 3000 

    Vnew = copy(V0_bis)  
    Vguess = copy(V0_bis)  
    policy_a_index = Array{Int64,2}(undef, Ns, Na) # Initialize the policy function index array
    tv = zeros(Na) # Temporary vector to store intermediate values

    for iter in 1:maxits
        for is in 1:Ns                     # Loop Over skills Today
            for ia in 1:Na                 # Loop Over assets Today
                tv = U[is, ia, :]' + par.beta * prob[is, :]' * Vguess[:, :] # Calculate the temporary value for each asset level tomorrow
                (Vnew[is, ia], policy_a_index[is, ia]) = findmax(tv[:]) # Find the maximum value and corresponding policy index
            end
        end
        if maximum(abs, Vguess .- Vnew) < tol
            println("Found solution after $iter iterations")
            break
        elseif iter == maxits
            println("No solution found after $iter iterations")
            break
        end
        Vguess = copy(Vnew)  # Update the guess
    end

    ##################

    return policy_a_index, Vnew
end

##############################################################################################################################
## BLOCK 2 ##############################################################################################################################
##############################################################################################################################

aiyagari = function(r,w,par,agrid,sgrid,prob,Vguess)
    Ns = length(sgrid)
    Na = length(agrid)

    # Call the VFI function and get the policy index 
    policy_a_index, Vnew = VFI(r, w, agrid, sgrid, Vguess, prob, par)

    ##### 1. Building the transition matrix  ####################
    # Build Q as a 4D array
    Q = zeros(Ns,Na,Ns,Na)

    ### TO FILL ####### 
    # Compute Q 
    # Transition Matrix
    for is in 1:Ns 
        for ia in 1:Na 
            ia_p = policy_a_index[is, ia] 
            for is_p in 1:Ns 
                Q[is, ia, is_p, ia_p] = prob[is, is_p] 
            end 
        end 
    end 
    ##################

    # Then reshape it if Q was 4D
    Q = reshape(Q, Ns*Na, Ns*Na)

    # Check that the rows sum to 1! 
    row_sums = sum(Q, dims=2) 

    tolerance = 1e-6 

    for i in 1:size(Q, 1) 
        if abs(row_sums[i] - 1) > tolerance 
            error("Sum of rows is $(row_sums[i])") 
        end 
    end 

    ###### 2. Computing the stable distribution  ###################################
    dist = ones(1, Ns*Na) / (Ns*Na);
    dist = get_stable_dist(dist, Q,par)

    # Check that the distribution vector sums to 1! 
    dist_sum = sum(dist) 

    tolerance = 1e-6 

    if abs(dist_sum - 1) > tolerance 
        error("Sum of distribution vector is $dist_sum") 
    end 

    # Reshape dist as a Ns x Na dimension (more readable) (don't have to)
    dist = reshape(dist, Ns, Na)

    ###### 3. Computing the aggregate #############################################
     # agg_a  
     agg_a = sum(dist*agrid) 

     # Aggregate supply of labor 
    agg_labor = sum(sgrid' * dist) 

    return agg_a, agg_labor, policy_a_index, Vnew, Q, dist 
end 


function get_stable_dist(invdist, P,par)
    for iter in 1:par.maxits
        invdist2 = invdist * P
        if maximum(abs, invdist2 .- invdist) < 1e-9
            println("Found solution after $iter iterations")
            return invdist2
        elseif iter == par.maxits
            error("No solution found after $iter iterations")
            return invdist
        end
        err = maximum(abs, invdist2 - invdist)
        invdist = invdist2
    end
end



##############################################################################################################################
## BLOCK 3 ##############################################################################################################################
##############################################################################################################################

## Intermediate, do not touch, before the 25/11 session 
# we just loop over 3 possible interest rates
find_equilibrium = function(par, agrid, sgrid, prob, Vguess) 

    # Initial bounds for r 
    r_low = -par.delta 
    r_high = 1/par.beta - 1 
    tol = 1e-3  # Tolerance 
    max_iter = 150  # Maximum iterations 
    ω = 0.5  # Weight for bisection step
    w = (1 - par.alpha) * (par.A * (par.alpha / ((r_low + r_high) / 2 + par.delta))^par.alpha)^(1 / (1 - par.alpha)) #Initial wage 

    for iter in 1:max_iter 
        # Midpoint interest rate 
        r_mid = (r_low + r_high) / 2 

        # Compute aggregate assets at r_mid 
        agg_a_mid, agg_labor_mid, policy_a_index, Vnew, Q, dist = aiyagari(r_mid, w, par, agrid, sgrid, prob, Vguess)
        # Compute capital implied by r_mid using the FOC
        K_mid = ((r_mid + par.delta) / par.alpha)^(1 / (par.alpha - 1)) * agg_labor_mid 

        # Excess supply (A - K) 
        excess = agg_a_mid - K_mid 
        println("After $iter iterations: r_mid = $r_mid, agg_a = $agg_a_mid, agg_labor = $agg_labor_mid,  K = $K_mid, excess = $excess") 

        # Check for convergence 
        if abs(excess) < tol 
            println("Equilibrium found: r = $r_mid after $iter iterations") 
            return r_mid, policy_a_index, w, Vnew, Q, dist 
        end 

        # Update bounds 
        if excess > 0 
            r_high = r_mid 
        else 
            r_low = r_mid 
        end 

        # Update r using weighted average 
        r_new = ω * r_mid + (1 - ω) * ((r_low + r_high) / 2) 

        # Update midpoint to weighted value 
        r_mid = r_new 

        # Update wage 
        w_new = (1 - par.alpha) * ((par.alpha * agg_labor_mid^(1 - par.alpha)) / (r_mid + par.delta))^(par.alpha / (1 - par.alpha)) * agg_labor_mid^(- par.alpha) 
        w = w_new     
    end 

    error("No equilibrium found after $max_iter iterations") 

end 

# Find equilibrium parameters 
r_star, policy_a_index, w, Vnew, Q, dist = find_equilibrium(par, agrid, sgrid, prob, Vguess) 

# plot result`
policy_a = Array{Float64,2}(undef,Ns,Na); 
for is in 1:Ns 
    policy_a[is,:] = agrid[policy_a_index[is,:]]  
end

policy_c = Array{Float64,2}(undef,Ns,Na);   
for is in 1:Ns 
    for ia in 1:Na 
        policy_c[is,ia] = (1+r_star)*agrid[ia] + sgrid[is]*w - policy_a[is,ia] 
    end  
end 

using Plots

vf_plot = plot(title="New Value Function for different skill levels", xlabel="Assets", ylabel="Value Function", legend=:bottomright) 

# Add each skill level to the plot with the correct label 
for is in 1:Ns 
    plot!(vf_plot, agrid, Vnew[is,:], label="Skill Level $is") 
end 

display(vf_plot)

# Policy function for consumption
plot(agrid, policy_c[:,:]', title="Policy Function for Consumption", xlabel="Assets", ylabel="Consumption", legend=false) 
 
# Policy function for assets 
plot(agrid, policy_a[:,:]', title="Policy Function for Assets", xlabel="Assets", ylabel="Assets Tomorrow", legend=false) 

# Create a plot 
p = plot(title="Stable Distribution for Skill Level", xlabel="Asset Levels (a)", ylabel="Density", legend=:topright) 

# Plot each row of the distribution matrix 
for s in 1:size(dist, 1) 
    p = plot!(agrid, dist[s, :], label="Skill Level $s") 
end 

# Display plot and transition matrix
display(p) 
Q


#plotting gini coefficient 
using Statistics 

function trapz(x, y) 
    return sum((y[2:end] .+ y[1:end-1]) .* (x[2:end] .- x[1:end-1]) / 2) 
end 

# Flatten the distribution matrix  
flat_dist = reshape(dist, :) 
sorted_indices = sortperm(agrid) 
sorted_dist = flat_dist[sorted_indices]

#sort by asset levels 
sorted_agrid = agrid[sorted_indices] 

# Compute cumulative shares 
cum_pop_share = cumsum(sorted_dist) / sum(sorted_dist) 
cum_wealth_share = cumsum(sorted_dist .* sorted_agrid) / sum(sorted_dist .* sorted_agrid) 

# Append point (1, 1) to the Lorenz curve 
cum_pop_share = [cum_pop_share; 1.0] 
cum_wealth_share = [cum_wealth_share; 1.0] 

# Compute the Gini coefficient using the trapezoidal rule 
gini_coeff = round(1 - trapz(cum_pop_share, cum_wealth_share); digits=3) 
println("Gini Coefficient: ", gini_coeff) 

# Plot Lorenz curve 
p = plot(cum_pop_share, cum_wealth_share, label="Lorenz Curve", xlabel="Cumulative Share of Population", ylabel="Cumulative Share of Wealth", title="Lorenz Curve (Gini = $gini_coeff)") 

# Add the line of perfect equality 
p = plot!([0, 1], [0, 1], label="Perfect Equality", linestyle=:dash) 

# Display plot 
display(p) 

