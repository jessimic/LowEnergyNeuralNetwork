year_array = np.linspace(2012, 2023, 1)
path1 = "/data/ana/LE/oscNext/pass2/pisa/flercnn/"
path2 = "/data/user/shiqiyu/fridge/processing/samples/oscNext/flercnn/"
set_names = ["orig data", "new data"]
set1=set_names[0]
set2 = set_names[1]

nu_keys = ['nue_cc', 'nue_nc', 'nuebar_cc', 'nuebar_nc', 'numu_cc', 'numu_nc', 'numubar_cc', 'numubar_nc', 'nutau_cc', 'nutau_nc', 'nutaubar_cc', 'nutaubar_nc']
saved_particles = []
particle = {}
particle[set_names[0]] = {}

for year in year_array:
    data_file_year = "oscNext_pisa_data_%s_flercnn.hdf5"%year
    set_files = [path1 + data_file_year, path2 + data_file_year]
    for input_file in input_file_list:
        print("Reading file %s"%input_file)
        f = h5py.File(input_file, "r")
    if "genie" in input_file:
        for nu_key in nu_keys:
            print("Reading %s"%nu_key)
            particle[set_names[0]][nu_key] = f[nu_key]
            saved_particles.append(nu_key)
    elif "muongun" in input_file:
        print("Reading muon")
        particle[set_names[0]]['muon'] = f['muon']
        saved_particles.append('muon')
    elif "noise" in input_file:
        print("Reading noise")
        particle[set_names[0]]['noise'] = f['noise']
        saved_particles.append('noise')
    else:
        print("Could not find simulation type in name!")
    #f.close()

if len(input_file_list2) > 0:
    particle[set_names[1]] = {}

    for input_file2 in input_file_list2:
        print("Reading file %s"%input_file2)
        f = h5py.File(input_file2, "r")
        if "genie" in input_file2:
            for nu_key in nu_keys:
                print("Reading %s"%nu_key)
                particle[set_names[1]][nu_key] = f[nu_key]
                assert nu_key in saved_particles, "Second files don't match neutrino keys/particles saved in first file"
                #saved_particles.append(nu_key)
        elif "muongun" in input_file2:
            print("Reading muon")
            particle[set_names[1]]['muon'] = f['muon']
            #saved_particles.append('muon')
            assert "muon" in saved_particles, "Second files don't match--muons saved here but not in first file"
        elif "noise" in input_file2:
            print("Reading noise")
            particle[set_names[1]]['noise'] = f['noise']
            #saved_particles.append('noise')
            assert "noise" in saved_particles, "Second files don't match--noise saved here but not in first file"
        else:
            print("Could not find simulation type in name!")

else:
    set_names = set_names[:1]


    
    for a_set in set_names:
        true[a_set] = {}
        print(set_files)

        for par_index in range(len(saved_particles)):
            particle_name = saved_particles[par_index]
            particle_here = particle[a_set][particle_name]

            weights[a_set] = particle_here['ReferenceWeight']
        
            if par_index == 0:
                true[a_set]['run_id'] = np.concatenate((true[a_set]['run_id'],np.array(particle_here['I3EventHeader.run_id'])))
                true[a_set]['subrun_id'] = np.concatenate((true[a_set]['subrun_id'], np.array(particle_here['I3EventHeader.sub_run_id'])))
                true[a_set]['event_id'] =  np.concatenate((true[a_set]['event_id'], np.array(particle_here['I3EventHeader.event_id'])))
            
            else:
                true[a_set]['subrun_id'] = np.concatenate((true[a_set]['subrun_id'], np.array(particle_here['I3EventHeader.sub_run_id'])))
                true[a_set]['event_id'] =  np.concatenate((true[a_set]['event_id'], np.array(particle_here['I3EventHeader.event_id'])))
                true[a_set]['deposited_energy'] =  np.concatenate((true[a_set]['deposited_energy'], deposited_energy))


        together = [str(i) + str(j) + str(k) for i, j, k in zip(true[a_set]['run_id'], true[a_set]['subrun_id'], true[a_set]['event_id'])]
        true[a_set]['full_ID'] = np.array(together,dtype=int )


    
    #Check length
    if len(true[set1]['full_ID']) != len(true[set2]['full_ID']):
        print("SETS AREN'T THE SAME SIZE!!!!"

    #Find shared events
    shared_events = len(set(true[set1]['full_ID']) & set(true[set2]['full_ID']))
    unique_set1 = len(true[set1]['full_ID']) - shared_events
    unique_set2 = len(true[set2]['full_ID']) - shared_events

    if unique_set1 > 0:
        print(set_files[0], " has ", sum(unique_set1), "unique events!")

    if unique_set2 > 0:
        print(set_files[1], " has ", sum(unique_set2), "unique events!")
