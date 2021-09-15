# Notices

Read about the assignment [here](./materials/assignment.pdf).

The report is expected to be compiled with [`tectonic`](https://tectonic-typesetting.github.io/en-US/) as follows:

```bash
tectonic -X compile report.tex
```

This project provides a [Julia](https://julialang.org) script. Make sure to use the project file (`Project.toml`) when running it:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. scripts/script.jl
```
