# This script takes data of a bimodal distribution
# and tries to optimize the parameters of the model
# distribution function by the MLE method

const DATA_PATH = "$(@__DIR__)/../data"
const TRACES_PATH = "$(@__DIR__)/../traces"
const PLOTS_PATH = "$(@__DIR__)/../plots"

println('\n', " "^4, "> Loading the packages...")

using DelimitedFiles # Delimited Files
using LaTeXStrings # LaTex strings
using Optim # Optimization
using Plots # Plotting
using Roots # Finding roots
using Zygote # Derivatives

# Use the GR backend for plots
gr()

# Change the default font for plots
default(fontfamily="Computer Modern", dpi=300, legend=:outerright)

println(" "^4, "> Loading the data...")

# Load the data (metallicities of globular clusters)
f = vec(readdlm("$(DATA_PATH)/all.dat"))

# Save the length of the data
N = length(f)

println(" "^4, "> Optimizing the negative log likelihood function...")

# Define the negative log likelihood function of the model
# (the model is a mixture of two normal distributions)
function nlml(θ::Vector{Float64})::Float64
    μ₁, σ₁, μ₂, σ₂, c = θ
    return -sum(
        @. log(
            c / σ₁ * exp(-(f - μ₁)^2 / (2 * σ₁^2)) +
            (1 - c) / σ₂ * exp(-(f - μ₂)^2 / (2 * σ₂^2))
        )
    )
end

# Define a set of initial parameters
θ₀ = [-1.5, 0.5, -0.5, 0.2, 0.5]

# Define the lower and upper boundaries, respectively
θₗ = [-Inf, 0, -Inf, 0, 0]
θᵤ = [Inf, Inf, Inf, Inf, 1]

# Optimize the negative log likelihood function
res = Optim.optimize(
    nlml,
    θ -> Zygote.gradient(nlml, θ)[1],
    θₗ,
    θᵤ,
    θ₀,
    Fminbox(LBFGS()),
    Optim.Options(
        show_trace=false,
        extended_trace=true,
        store_trace=true,
    );
    inplace=false,
)

# Unpack the results
μ₁, σ₁, μ₂, σ₂, c = res.minimizer
L₀ = res.minimum

# Save the trace and the results
open("$(TRACES_PATH)/all.trace", "w") do io
    println(io, res.trace)
    println(
        io,
        " * Parameters:\n",
        "    μ₁ = $(μ₁)\n",
        "    σ₁ = $(σ₁)\n",
        "    μ₂ = $(μ₂)\n",
        "    σ₂ = $(σ₂)\n",
        "    c = $(c)\n"
    )
    show(io, res)
end

# Print results
println(
    '\n',
    " "^6, "μ₁ = $(μ₁)\n",
    " "^6, "σ₁ = $(σ₁)\n",
    " "^6, "μ₂ = $(μ₂)\n",
    " "^6, "σ₂ = $(σ₂)\n",
    " "^6, "c = $(c)\n",
    '\n',
    " "^6, "* The trace `all.trace` is saved. *\n"
)

println(" "^4, "> Plotting the histogram...")

# Define the histogram's step
Δf = 0.1

# Calculate the left border of the histogram
lb = Δf * floor(minimum(f) / Δf)

# Prepare a range for the model distribution
r = range(lb, 0; length=1000)

# Calculate the model distribution
φ =  @. 1 / sqrt(2 * π) * (
    c / σ₁ * exp(-(r - μ₁)^2 / (2 * σ₁^2)) +
    (1 - c) / σ₂ * exp(-(r - μ₂)^2 / (2 * σ₂^2))
)

# Create a histogram of observed data
p = histogram(
    f;
    label="Наблюдаемые данные",
    xlabel="Показатель металличности",
    ylabel="Число скоплений",
    bins=range(lb, 0; step=Δf),
    color="#80cdfd",
);
plot!(p, minorticks=5, yticks=range(0, ceil(Int, ylims(p)[2]); step=5))

# Add the scaled model data to the plot
plot!(p, r, N * Δf * φ; label="Модельная функция", lw=1.5);

# Save the figure
savefig(p, "$(PLOTS_PATH)/histogram.pdf")

# Print about the figure
println('\n', " "^6, "* The figure `histogram.pdf` is saved. *", '\n')

# Calculate the minimum of the negative log likelihood function
# while one of the parameters is frozen
function nlml_frozen(
    idx::Int,
    value::Float64,
    θ₀::Vector{Float64},
    θₗ::Vector{Float64},
    θᵤ::Vector{Float64}
)::Float64
    # Exclude the frozen parameter from the active ones
    θ₀ = [θ₀[1:idx - 1]; θ₀[idx + 1:end]]
    θₗ = [θₗ[1:idx - 1]; θₗ[idx + 1:end]]
    θᵤ = [θᵤ[1:idx - 1]; θᵤ[idx + 1:end]]

    # Recreate the function, handling a frozen parameter
    function nlml_frozen_inner(θ::Vector{Float64})
        μ₁, σ₁, μ₂, σ₂, c = [θ[1:idx - 1]; value; θ[idx:end]]
        return -sum(
            @. log(
                c / σ₁ * exp(-(f - μ₁)^2 / (2 * σ₁^2)) +
                (1 - c) / σ₂ * exp(-(f - μ₂)^2 / (2 * σ₂^2))
            )
        )
    end

    # Optimize the new negative log likelihood function
    res = Optim.optimize(
        nlml_frozen_inner,
        θ -> Zygote.gradient(nlml_frozen_inner, θ)[1],
        θₗ,
        θᵤ,
        θ₀,
        Fminbox(LBFGS()),
        Optim.Options(
            show_trace=false,
            extended_trace=true,
            store_trace=true,
        );
        inplace=false,
    )

    # Unpack the results
    μ₁, σ₁, μ₂, σ₂, c = [res.minimizer[1:idx - 1]; value; res.minimizer[idx:end]]

    # Create a directory for results if it doesn't exist
    if !isdir("$(TRACES_PATH)/$(idx)")
        mkdir("$(TRACES_PATH)/$(idx)")
    end

    # Save the trace and the results
    open("$(TRACES_PATH)/$(idx)/$(value).trace", "w") do io
        println(io, res.trace)
        println(
            io,
            " * Parameters:\n",
            "    μ₁ = $(μ₁)\n",
            "    σ₁ = $(σ₁)\n",
            "    μ₂ = $(μ₂)\n",
            "    σ₂ = $(σ₂)\n",
            "    c = $(c)\n"
        )
        show(io, res)
    end

    return res.minimum
end

println(" "^4, "> Calculating the profile of μ₁...")

# Create an alias function for the frozen parameter μ₁
nlml_μ₁(μ₁) = nlml_frozen(1, μ₁, θ₀, θₗ, θᵤ)

# Find the first root
μ₁₊₋ₕ = find_zero((μ₁) -> nlml_μ₁(μ₁) - L₀ - 0.5, μ₁)

# Find the second one nearby; calculate the confidence intervals
μ₁₋ₕ = μ₁₊ₕ = σμ₁₋ₕ = σμ₁₊ₕ = 0
if μ₁ - μ₁₊₋ₕ > 0
    μ₁₋ₕ = μ₁₊₋ₕ
    σμ₁₋ₕ = abs(μ₁ - μ₁₋ₕ)
    μ₁₊ₕ = find_zero((μ₁) -> nlml_μ₁(μ₁) - L₀ - 0.5, μ₁ + σμ₁₋ₕ)
    σμ₁₊ₕ = abs(μ₁ - μ₁₊ₕ)
else
    μ₁₊ₕ = μ₁₊₋ₕ
    σμ₁₊ₕ = abs(μ₁ - μ₁₊ₕ)
    μ₁₋ₕ = find_zero((μ₁) -> nlml_μ₁(μ₁) - L₀ - 0.5, μ₁ - σμ₁₊ₕ)
    σμ₁₋ₕ = abs(μ₁ - μ₁₋ₕ)
end

# Print the results
println(
    '\n',
    " "^6, "μ₁ = $(μ₁)\n",
    " "^6, "μ₁₋ₕ = $(μ₁₋ₕ)\n",
    " "^6, "μ₁₊ₕ = $(μ₁₊ₕ)\n",
    " "^6, "σμ₁₋ₕ = $(σμ₁₋ₕ)\n",
    " "^6, "σμ₁₊ₕ = $(σμ₁₊ₕ)\n",
    '\n',
    " "^6, "* Traces saved in the traces/1 directory. *",
    '\n',
)

println(" "^4, "> Plotting the profile of μ₁...")

# Plot the profile of μ₁
p = plot(nlml_μ₁, μ₁₋ₕ - σμ₁₋ₕ, μ₁₊ₕ + σμ₁₊ₕ; label="Профиль", xlabel=L"\mu_1", ylabel=L"L_p(\mu_1)");

# Add vertical lines to the plot
plot!(p, [μ₁₋ₕ, μ₁₋ₕ], [L₀ - 0.2, L₀ + 0.5]; label=L"\mu_{1-}", linestyle=:dash);
plot!(p, [μ₁, μ₁], [L₀ - 0.2, L₀]; label=L"\mu_{1_0}", linestyle=:dash);
plot!(p, [μ₁₊ₕ, μ₁₊ₕ], [L₀ - 0.2, L₀ + 0.5]; label=L"\mu_{1+}", linestyle=:dash);

# Add points to the plot
scatter!(p, [μ₁₋ₕ,], [L₀ + 0.5,]; label="");
scatter!(p, [μ₁,], [L₀,]; label="");
scatter!(p, [μ₁₊ₕ,], [L₀ + 0.5,]; label="");

# Add horizontal lines to the plot
hline!(p, [L₀ + 0.5,]; label=L"L_0 + 1/2", linestyle=:dash);
hline!(p, [L₀,]; label=L"L_0", linestyle=:dash)

# Add annotations to the plot
annotate!(p, [ (v + 0.01, L₀ - 0.1, text("$(round(v; digits=2))", 8, "Computer Modern")) for v in [μ₁₋ₕ, μ₁, μ₁₊ₕ] ]);
annotate!(
    p,
    [
        (-1.62, L₀ + 0.06, text(L"L_0", 8, "Computer Modern")),
        (-1.61, L₀ + 0.5 + 0.06, text(L"L_0 + 1/2", 8, "Computer Modern")),
    ]
);

# Save the figure
savefig(p, "$(PLOTS_PATH)/μ₁.pdf")

# Print about the figure
println('\n', " "^6, "* The figure `μ₁.pdf` is saved. *", '\n')

println(" "^4, "> Calculating the profile of σ₁...")

# Create an alias function for the frozen parameter σ₁
nlml_σ₁(σ₁) = nlml_frozen(2, σ₁, θ₀, θₗ, θᵤ)

# Find the first root
σ₁₊₋ₕ = find_zero((σ₁) -> nlml_σ₁(σ₁) - L₀ - 0.5, σ₁)

# Find the second one nearby; calculate the confidence intervals
σ₁₋ₕ = σ₁₊ₕ = σσ₁₋ₕ = σσ₁₊ₕ = 0
if σ₁ - σ₁₊₋ₕ > 0
    σ₁₋ₕ = σ₁₊₋ₕ
    σσ₁₋ₕ = abs(σ₁ - σ₁₋ₕ)
    σ₁₊ₕ = find_zero((σ₁) -> nlml_σ₁(σ₁) - L₀ - 0.5, σ₁ + σσ₁₋ₕ)
    σσ₁₊ₕ = abs(σ₁ - σ₁₊ₕ)
else
    σ₁₊ₕ = σ₁₊₋ₕ
    σσ₁₊ₕ = abs(σ₁ - σ₁₊ₕ)
    σ₁₋ₕ = find_zero((σ₁) -> nlml_σ₁(σ₁) - L₀ - 0.5, σ₁ - σσ₁₊ₕ)
    σσ₁₋ₕ = abs(σ₁ - σ₁₋ₕ)
end

# Print the results
println(
    '\n',
    " "^6, "σ₁ = $(σ₁)\n",
    " "^6, "σ₁₋ₕ = $(σ₁₋ₕ)\n",
    " "^6, "σ₁₊ₕ = $(σ₁₊ₕ)\n",
    " "^6, "σσ₁₋ₕ = $(σσ₁₋ₕ)\n",
    " "^6, "σσ₁₊ₕ = $(σσ₁₊ₕ)\n",
    '\n',
    " "^6, "* Traces saved in the traces/2 directory. *",
    '\n',
)

println(" "^4, "> Plotting the profile of σ₁...")

# Plot the profile of σ₁
p = plot(nlml_σ₁, σ₁₋ₕ - σσ₁₋ₕ, σ₁₊ₕ + σσ₁₊ₕ; label="Профиль", xlabel=L"\sigma_1", ylabel=L"L_p(\sigma_1)");

# Add vertical lines to the plot
plot!(p, [σ₁₋ₕ, σ₁₋ₕ], [L₀ - 0.2, L₀ + 0.5]; label=L"\sigma_{1-}", linestyle=:dash);
plot!(p, [σ₁, σ₁], [L₀ - 0.2, L₀]; label=L"\sigma_{1_0}", linestyle=:dash);
plot!(p, [σ₁₊ₕ, σ₁₊ₕ], [L₀ - 0.2, L₀ + 0.5]; label=L"\sigma_{1+}", linestyle=:dash);

# Add points to the plot
scatter!(p, [σ₁₋ₕ,], [L₀ + 0.5,]; label="");
scatter!(p, [σ₁,], [L₀,]; label="");
scatter!(p, [σ₁₊ₕ,], [L₀ + 0.5,]; label="");

# Add horizontal lines to the plot
hline!(p, [L₀ + 0.5,]; label=L"L_0 + 1/2", linestyle=:dash);
hline!(p, [L₀,]; label=L"L_0", linestyle=:dash)

# Add annotations to the plot
annotate!(p, [ (v + 0.007, L₀ - 0.1, text("$(round(v; digits=2))", 8, "Computer Modern")) for v in [σ₁₋ₕ, σ₁, σ₁₊ₕ] ]);
annotate!(
    p,
    [
        (0.322, L₀ + 0.06, text(L"L_0", 8, "Computer Modern")),
        (0.330, L₀ + 0.5 + 0.06, text(L"L_0 + 1/2", 8, "Computer Modern")),
    ]
);

# Save the figure
savefig(p, "$(PLOTS_PATH)/σ₁.pdf")

# Print about the figure
println('\n', " "^6, "* The figure `σ₁.pdf` is saved. *", '\n')

println(" "^4, "> Calculating the profile of μ₂...")

# Create an alias function for the frozen parameter μ₂
nlml_μ₂(μ₂) = nlml_frozen(3, μ₂, θ₀, θₗ, θᵤ)

# Find the first root
μ₂₊₋ₕ = find_zero((μ₂) -> nlml_μ₂(μ₂) - L₀ - 0.5, μ₂)

# Find the second one nearby; calculate the confidence intervals
μ₂₋ₕ = μ₂₊ₕ = σμ₂₋ₕ = σμ₂₊ₕ = 0
if μ₂ - μ₂₊₋ₕ > 0
    μ₂₋ₕ = μ₂₊₋ₕ
    σμ₂₋ₕ = abs(μ₂ - μ₂₋ₕ)
    μ₂₊ₕ = find_zero((μ₂) -> nlml_μ₂(μ₂) - L₀ - 0.5, μ₂ + σμ₂₋ₕ)
    σμ₂₊ₕ = abs(μ₂ - μ₂₊ₕ)
else
    μ₂₊ₕ = μ₂₊₋ₕ
    σμ₂₊ₕ = abs(μ₂ - μ₂₊ₕ)
    μ₂₋ₕ = find_zero((μ₂) -> nlml_μ₂(μ₂) - L₀ - 0.5, μ₂ - σμ₂₊ₕ)
    σμ₂₋ₕ = abs(μ₂ - μ₂₋ₕ)
end

# Print the results
println(
    '\n',
    " "^6, "μ₂ = $(μ₂)\n",
    " "^6, "μ₂₋ₕ = $(μ₂₋ₕ)\n",
    " "^6, "μ₂₊ₕ = $(μ₂₊ₕ)\n",
    " "^6, "σμ₂₋ₕ = $(σμ₂₋ₕ)\n",
    " "^6, "σμ₂₊ₕ = $(σμ₂₊ₕ)\n",
    '\n',
    " "^6, "* Traces saved in the traces/3 directory. *",
    '\n',
)

println(" "^4, "> Plotting the profile of μ₂...")

# Plot the profile of μ₂
p = plot(nlml_μ₂, μ₂₋ₕ - σμ₂₋ₕ, μ₂₊ₕ + σμ₂₊ₕ; label="Профиль", xlabel=L"\mu_2", ylabel=L"L_p(\mu_2)");

# Add vertical lines to the plot
plot!(p, [μ₂₋ₕ, μ₂₋ₕ], [L₀ - 0.2, L₀ + 0.5]; label=L"\mu_{2-}", linestyle=:dash);
plot!(p, [μ₂, μ₂], [L₀ - 0.2, L₀]; label=L"\mu_{2_0}", linestyle=:dash);
plot!(p, [μ₂₊ₕ, μ₂₊ₕ], [L₀ - 0.2, L₀ + 0.5]; label=L"\mu_{2+}", linestyle=:dash);

# Add points to the plot
scatter!(p, [μ₂₋ₕ,], [L₀ + 0.5,]; label="");
scatter!(p, [μ₂,], [L₀,]; label="");
scatter!(p, [μ₂₊ₕ,], [L₀ + 0.5,]; label="");

# Add horizontal lines to the plot
hline!(p, [L₀ + 0.5,]; label=L"L_0 + 1/2", linestyle=:dash);
hline!(p, [L₀,]; label=L"L_0", linestyle=:dash)

# Add annotations to the plot
annotate!(p, [ (v + 0.011, L₀ - 0.1, text("$(round(v; digits=2))", 8, "Computer Modern")) for v in [μ₂₋ₕ, μ₂, μ₂₊ₕ] ]);
annotate!(
    p,
    [
        (-0.65, L₀ + 0.06, text(L"L_0", 8, "Computer Modern")),
        (-0.638, L₀ + 0.5 + 0.06, text(L"L_0 + 1/2", 8, "Computer Modern")),
    ]
);

# Save the figure
savefig(p, "$(PLOTS_PATH)/μ₂.pdf")

# Print about the figure
println('\n', " "^6, "* The figure `μ₂.pdf` is saved. *", '\n')

println(" "^4, "> Calculating the profile of σ₂...")

# Create an alias function for the frozen parameter σ₂
nlml_σ₂(σ₂) = nlml_frozen(4, σ₂, θ₀, θₗ, θᵤ)

# Find the first root
σ₂₊₋ₕ = find_zero((σ₂) -> nlml_σ₂(σ₂) - L₀ - 0.5, σ₂)

# Find the second one nearby; calculate the confidence intervals
σ₂₋ₕ = σ₂₊ₕ = σσ₂₋ₕ = σσ₂₊ₕ = 0
if σ₂ - σ₂₊₋ₕ > 0
    σ₂₋ₕ = σ₂₊₋ₕ
    σσ₂₋ₕ = abs(σ₂ - σ₂₋ₕ)
    σ₂₊ₕ = find_zero((σ₂) -> nlml_σ₂(σ₂) - L₀ - 0.5, σ₂ + σσ₂₋ₕ)
    σσ₂₊ₕ = abs(σ₂ - σ₂₊ₕ)
else
    σ₂₊ₕ = σ₂₊₋ₕ
    σσ₂₊ₕ = abs(σ₂ - σ₂₊ₕ)
    σ₂₋ₕ = find_zero((σ₂) -> nlml_σ₂(σ₂) - L₀ - 0.5, σ₂ - σσ₂₊ₕ)
    σσ₂₋ₕ = abs(σ₂ - σ₂₋ₕ)
end

# Print the results
println(
    '\n',
    " "^6, "σ₂ = $(σ₂)\n",
    " "^6, "σ₂₋ₕ = $(σ₂₋ₕ)\n",
    " "^6, "σ₂₊ₕ = $(σ₂₊ₕ)\n",
    " "^6, "σσ₂₋ₕ = $(σσ₂₋ₕ)\n",
    " "^6, "σσ₂₊ₕ = $(σσ₂₊ₕ)\n",
    '\n',
    " "^6, "* Traces saved in the traces/4 directory. *",
    '\n',
)

println(" "^4, "> Plotting the profile of σ₂...")

# Plot the profile of σ₂
p = plot(nlml_σ₂, σ₂₋ₕ - σσ₂₋ₕ, σ₂₊ₕ + σσ₂₊ₕ; label="Профиль", xlabel=L"\sigma_2", ylabel=L"L_p(\sigma_2)");

# Add vertical lines to the plot
plot!(p, [σ₂₋ₕ, σ₂₋ₕ], [L₀ - 0.2, L₀ + 0.5]; label=L"\sigma_{2-}", linestyle=:dash);
plot!(p, [σ₂, σ₂], [L₀ - 0.2, L₀]; label=L"\sigma_{2_0}", linestyle=:dash);
plot!(p, [σ₂₊ₕ, σ₂₊ₕ], [L₀ - 0.2, L₀ + 0.5]; label=L"\sigma_{2+}", linestyle=:dash);

# Add points to the plot
scatter!(p, [σ₂₋ₕ,], [L₀ + 0.5,]; label="");
scatter!(p, [σ₂,], [L₀,]; label="");
scatter!(p, [σ₂₊ₕ,], [L₀ + 0.5,]; label="");

# Add horizontal lines to the plot
hline!(p, [L₀ + 0.5,]; label=L"L_0 + 1/2", linestyle=:dash);
hline!(p, [L₀,]; label=L"L_0", linestyle=:dash)

# Add annotations to the plot
annotate!(p, [ (v + 0.007, L₀ - 0.1, text("$(round(v; digits=2))", 8, "Computer Modern")) for v in [σ₂₋ₕ, σ₂, σ₂₊ₕ] ]);
annotate!(
    p,
    [
        (0.160, L₀ + 0.06, text(L"L_0", 8, "Computer Modern")),
        (0.168, L₀ + 0.5 + 0.06, text(L"L_0 + 1/2", 8, "Computer Modern")),
    ]
);

# Save the figure
savefig(p, "$(PLOTS_PATH)/σ₂.pdf")

# Print about the figure
println('\n', " "^6, "* The figure `σ₂.pdf` is saved. *", '\n')

println(" "^4, "> Calculating the profile of c...")

# Create an alias function for the frozen parameter c
nlml_c(c) = nlml_frozen(5, c, θ₀, θₗ, θᵤ)

# Find the first root
c₊ₕ = find_zero((c) -> nlml_c(c) - L₀ - 0.5, (c, 1.0))

# Find the second one nearby; calculate the confidence intervals
σc₊ₕ = abs(c - c₊ₕ)
c₋ₕ = find_zero((c) -> nlml_c(c) - L₀ - 0.5, c - σc₊ₕ)
σc₋ₕ = abs(c - c₋ₕ)

# Print the results
println(
    '\n',
    " "^6, "c = $(c)\n",
    " "^6, "c₋ₕ = $(c₋ₕ)\n",
    " "^6, "c₊ₕ = $(c₊ₕ)\n",
    " "^6, "σc₋ₕ = $(σc₋ₕ)\n",
    " "^6, "σc₊ₕ = $(σc₊ₕ)\n",
    '\n',
    " "^6, "* Traces saved in the traces/5 directory. *",
    '\n',
)

println(" "^4, "> Plotting the profile of c...")

# Plot the profile of c
p = plot(nlml_c, c₋ₕ - σc₋ₕ, c₊ₕ + σc₊ₕ; label="Профиль", xlabel=L"c", ylabel=L"L_p(c)");

# Add vertical lines to the plot
plot!(p, [c₋ₕ, c₋ₕ], [L₀ - 0.2, L₀ + 0.5]; label=L"c_{-}", linestyle=:dash);
plot!(p, [c, c], [L₀ - 0.2, L₀]; label=L"c_0", linestyle=:dash);
plot!(p, [c₊ₕ, c₊ₕ], [L₀ - 0.2, L₀ + 0.5]; label=L"c_{+}", linestyle=:dash);

# Add points to the plot
scatter!(p, [c₋ₕ,], [L₀ + 0.5,]; label="");
scatter!(p, [c,], [L₀,]; label="");
scatter!(p, [c₊ₕ,], [L₀ + 0.5,]; label="");

# Add horizontal lines to the plot
hline!(p, [L₀ + 0.5,]; label=L"L_0 + 1/2", linestyle=:dash);
hline!(p, [L₀,]; label=L"L_0", linestyle=:dash)

# Add annotations to the plot
annotate!(p, [ (v + 0.01, L₀ - 0.1, text("$(round(v; digits=2))", 8, "Computer Modern")) for v in [c₋ₕ, c, c₊ₕ] ]);
annotate!(
    p,
    [
        (0.64, L₀ + 0.06, text(L"L_0", 8, "Computer Modern")),
        (0.65, L₀ + 0.5 + 0.06, text(L"L_0 + 1/2", 8, "Computer Modern")),
    ]
);

# Save the figure
savefig(p, "$(PLOTS_PATH)/c.pdf")

# Print about the figure
println('\n', " "^6, "* The figure `c.pdf` is saved. *", '\n')

println(" "^4, "> The script finished.", '\n')
