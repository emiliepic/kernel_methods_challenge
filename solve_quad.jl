using JuMP
using Ipopt
using CPLEX
using CSV
using DataFrames
using JSON

# Définir les données
name_json = ARGS[1]
alpha_path = ARGS[2]

json_file = open(name_json)
data = JSON.parse(json_file)
close(json_file)
n = data["n"]
C = data["C"]
y = data["y"]
K = data["K"]


# Créer un modèle d'optimisation quadratique
model = Model(Ipopt.Optimizer)

# Définir les variables alpha
@variable(model, alpha[1:n] >= 0)

# Définir la fonction objectif
@objective(model, Min, -sum(alpha) + 0.5 * sum(alpha[i] * alpha[j] * y[i] * y[j] * K[i][j] for i in 1:n, j in 1:n))

# Ajouter la contrainte de somme des α_i*y_i
@constraint(model, sum(alpha[i] * y[i] for i in 1:n) == 0)

# Ajouter les bornes sur les variables α
@constraint(model, alpha .<= C)

# Résoudre le problème d'optimisation
optimize!(model)

# Extraire la solution
alpha_opt = value.(alpha)

# save in a csv file
CSV.write(alpha_path, DataFrame(alpha_opt=alpha_opt))