from icecube import NewNuFlux
def weight_frame(frame, nfiles=1, isNuGen=False):
    lowe_flux_service = NewNuFlux.makeFlux("IPhonda2014_spl_solmin")
    highe_flux_service = NewNuFlux.makeFlux("honda2006")
    if isNuGen: gen_ratio=0.5
    else: gen_ratio=0.7
    mc_weights = frame['I3MCWeightDict']
    true_neutrino = dataclasses.get_most_energetic_neutrino(frame['I3MCTree'])
    true_energy = mc_weights['PrimaryNeutrinoEnergy']
    true_zenith = true_neutrino.dir.zenith
    if true_neutrino.energy < 10000: flux_service = lowe_flux_service
    else: flux_service = highe_flux_service
    nue, numu = I3Particle.ParticleType.NuE, I3Particle.ParticleType.NuMu
    nu_nubar_genratio = gen_ratio
    if true_neutrino.pdg_encoding < 0:
        nue, numu = I3Particle.ParticleType.NuEBar, I3Particle.ParticleType.NuMuBar
        nu_nubar_genratio = 1-nu_nubar_genratio
    if np.abs(true_neutrino.pdg_encoding) == 12:
        flux = flux_service.getFlux(nue, true_energy, numpy.cos(true_zenith))
    if np.abs(true_neutrino.pdg_encoding) == 14:
        flux = flux_service.getFlux(numu, true_energy, numpy.cos(true_zenith))
    one_weight = mc_weights['OneWeight']
    n_events = mc_weights['NEvents']
    norm = (1.0 / (n_events * nfiles * nu_nubar_genratio))
    w = norm * one_weight * flux
    return w
