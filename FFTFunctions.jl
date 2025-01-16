#=

    In this module we write several functions useful for computing fourier transforms, and extracting useful things from them

=#

module  FourierFunctions
using FFTW, ..FourierFitGSL, ..FourierFitGSL_Derivs, GSL

# find local maxima of a signal
function findlocalmaxima(signal::Vector{Float64})
    inds = Int[]
    if length(signal)>1
        if signal[1]>signal[2]
            push!(inds,1)
        end
        for i=2:length(signal)-1
            if signal[i-1]<signal[i]>signal[i+1]
                push!(inds,i)
            end
        end
        if signal[end]>signal[end-1]
            push!(inds,length(signal))
        end
    end
    inds
end

# computes FFT of a real signal
function compute_real_FFT(signal::Vector{Float64}, fs::Float64, nPoints::Int64)
    F = rfft(signal)
    F_freqs = rfftfreq(nPoints, fs)
    return F, F_freqs
end

# this function takes as input some real FFT, and outputs the N most dominant harmonic frequenices — nFreqs==-1 for all peak frequencies
function extract_frequencies!(F::Vector{ComplexF64}, F_freqs::Frequencies{Float64}, nFreqs::Int64, exclude_zero::Bool)
    # finding local maxima of the fourier transform
    dominant_indices = findlocalmaxima(abs.(F))

    # extract peaks and the corresponding frequencies
    peaks_freqs = F_freqs[dominant_indices]
    peaks_F_vals = abs.(F[dominant_indices])

    # now sort frequencies in decreasing order of the height of their respective peaks
    perm = sortperm(peaks_F_vals, rev=true); 
    peaks_freqs .= peaks_freqs[perm]
    peaks_F_vals .= peaks_F_vals[perm]

    # extract most dominant frequencies
    if nFreqs==-1
        ordered_freqs = peaks_freqs;  ordered_F_vals = peaks_F_vals;
    else
        ordered_freqs = peaks_freqs[1:nFreqs];  ordered_F_vals = peaks_F_vals[1:nFreqs];
    end

    # exclude zero frequency
    if exclude_zero==true
        for i in eachindex(ordered_freqs)
            if ordered_freqs[i]==0.0
                println(i)
                deleteat!(ordered_freqs, i); 
                deleteat!(ordered_F_vals, i);
                
                if nFreqs!=-1
                    append!(ordered_freqs, peaks_freqs[nFreqs+1])
                    append!(ordered_F_vals, peaks_F_vals[nFreqs+1])
                end
                break
            end
        end
    end

    return 2π .* ordered_freqs, ordered_F_vals
end
end