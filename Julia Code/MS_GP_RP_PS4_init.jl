using Distributions, LinearAlgebra

##Define Parameters
beta = 0.95 # Discount factor
rho = 0.9; # persistence of the AR(1) process
sigma = 0.2; # Standart deviation of the AR(1) process
Ns = 5; # Number of states of the Markov Chain discretization 
b = -0.2 # borrowing limit
r = 0.025 # interest rate 
w = 1  # wage 
mu = 2 # risk aversion. CRRA             

##Defining the Grid for the Endogenous State Variable: Capital
Na = 300;
amax =  60;
agrid = collect(range(b, length = Na, stop = amax));


##Defining the Grid for the Exogenous State Variable: Technology Shock
function tauchen(mean, sd, rho, num_states; q=3)

    uncond_sd = sd/sqrt(1-rho^2)
    y = range(-q*uncond_sd, stop = q*uncond_sd, length = num_states)
    d = y[2]-y[1]

    Pi = zeros(num_states,num_states)

    for row = 1:num_states
      # end points
          Pi[row,1] = cdf(Normal(),(y[1] - rho*y[row] + d/2)/sd)
          Pi[row,num_states] = 1 - cdf(Normal(), (y[num_states] - rho*y[row] - d/2)/sd)

      # middle columns
          for col = 2:num_states-1
              Pi[row, col] = (cdf(Normal(),(y[col] - rho*y[row] + d/2) / sd) -
                             cdf(Normal(),(y[col] - rho*y[row] - d/2) / sd))
          end
    end

  yy = y .+ mean # center process around its mean

  Pi = Pi./sum(Pi, dims = 2) # renormalize

  return Pi, yy
end 

prob = tauchen(0, sigma, rho, Ns)[1];
logs = tauchen(0, sigma, rho, Ns)[2];
sgrid = exp.(logs);



## Build the 3-Dimensional Contemporaneous Utility Grid for the System
# Initialize the 3D Array
U = zeros(Ns,Na,Na)

## Build 3D utility array ############################################################################################################
for is in 1:Ns                     # Loop Over skills Today
    for ia in 1:Na                 # Loop Over assets Today
        for ia_p in 1:Na           # Loop Over assets Tomorrow
            c = sgrid[is] * w + (1 + r) * agrid[ia] - agrid[ia_p]  # Consumption
            if c <= 0
                U[is, ia, ia_p] = -1e6  # Penalize infeasible consumption
            else
                if mu == 1
                    U[is, ia, ia_p] = log(c) # Log utility for CRRA with mu = 1
                else
                    U[is, ia, ia_p] = (c^(1 - mu) - 1) / (1 - mu) # CRRA utility
                end
            end
        end
    end
end
##################################################################################################################


##Value Function Iteration

#Initial Guess of the Value Function
V0 = zeros(Ns,Na);

tol = 1e-4;
its = 0;
maxits = 3000; # Define the maximum number of iterations
Vnew = copy(V0);  # The new value function I obtain after an iteration
Vguess = copy(V0);  # the  value function from which I start in each new iteration
policy_a_index = Array{Int64,2}(undef,Ns,Na);
tv = zeros(Na)

#Second upgraded initial guess 
### Expected Value Function ############################################################################################################
function compute_expected_value(V, prob)
    Ns, Na = size(V)
    EV = zeros(Ns, Na)
    for is in 1:Ns
        for ia in 1:Na
            EV[is, ia] = sum(prob[is, :] .* V[:, ia])
        end
    end
    return EV
end
##################################################################################################################

#Calculate the Guess of the Expected Value Function

###### TO FILL ############################################################################################################
 for iter in 1:maxits
     for is in 1:Ns                     # Loop Over skills Today
         for ia in 1:Na                 # Loop Over assets Today
             # Compute tv 
             for ia_p in 1:Na           # Loop Over assets Tomorrow
                tv[ia_p] = U[is, ia, ia_p] + beta * sum(prob[is, js] * Vguess[js, ia_p] for js in 1:Ns)
             end 

             #max value and index
             Vnew[is,ia], policy_a_index[is,ia] = findmax(tv)
         end
     end
     if maximum(abs,Vguess.-Vnew) < tol
         println("Found solution after $iter iterations")
     return nothing
     elseif iter==maxits
         println("No solution found after $iter iterations")
     return nothing
     end
     err = maximum(abs,Vguess.-Vnew)      # compute the error 
     Vguess .= Vnew     # update guess 
     println("#iter = $iter, εᵥ = $err")
 end
##################################################################################################################




##Policy function for assets
policy_a = Array{Float64,2}(undef,Ns,Na);

for is in 1:Ns
policy_a[is,:] = agrid[policy_a_index[is,:]] 
end


##Policy function for consumption
policy_c = Array{Float64,2}(undef,Ns,Na) 
###### TO FILL ############################################################################################################
# Recover policy function for consumption 
policy_c = Array{Float64, 2}(undef, Ns, Na)
for is in 1:Ns
    for ia in 1:Na
        policy_c[is, ia] = sgrid[is] * w + (1 + r) * agrid[ia] - policy_a[is, ia]
    end
end

##################################################################################################################


##Plotting the results
using Plots
plot(agrid,Vnew[:,:]')
plot(agrid,policy_c[:,:]')
plot(agrid,policy_a[:,:]')
