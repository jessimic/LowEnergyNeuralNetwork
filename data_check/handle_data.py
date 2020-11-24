############################
# Important functions used when processing data for neural network training
#   CutMask: Creates dict of masks based on set labels isTrack and isCC
#   Shuffler: shuffles features_DC, features_IC, and labels, keeping events together
#   SplitTrainTest: Splits features_DC, features_IC, and labels into 2 or 3 sets (train, test, validate (optional))
#############################

import numpy as np
import math

def CutMask(set_labels):
    """Creates dictionary full of possible cut masks
    NOTE: cut name is the type of events you want to KEEP
    Currently outputs cuts: track, cascade, CC, NC, track CC, track NC, cascade CC, cascade NC, all
    Receives:
        set_label: labels array, expects [event number, 12]
    Labels order: [energy, zenith, azimyth, time, x, y, z, track length, isTrack, flavor, isAnti, isCC]
    Outputs:
        mask: dict with all masks possible
    """
    isTrack = np.array(set_labels[:,8])
    isCC = np.array(set_labels[:,11])

    mask = {}
    mask['track'] = isTrack==1 
    mask['cascade'] = isTrack==0 
    mask['CC'] = isCC==1 
    mask['NC'] = isCC==0
    mask['track CC'] = np.logical_and( mask['track'], mask['CC'] )
    mask['track NC'] = np.logical_and( mask['track'], mask['NC'] )
    mask['cascade CC'] = np.logical_and( mask['cascade'], mask['CC'] )
    mask['cascade NC'] = np.logical_and( mask['cascade'], mask['NC'] )
    mask['all'] = np.logical_or( isTrack==1, isTrack==0)
    
    return mask

def VertexMask(set_labels,azimuth_index=2,track_index=7,max_track=1.0):

    # Position of String 36 for origin
    x_origin = 46.290000915527344
    y_origin = -34.880001068115234

    # Load true labels
    theta = set_labels[:,1] #zenith
    phi = set_labels[:,azimuth_index] #azimuth
    x_start = set_labels[:,4]
    y_start = set_labels[:,5]
    z_start = set_labels[:,6]
    track_length = set_labels[:,track_index]*max_track
    n_x = np.sin(theta)*np.cos(phi)
    n_y = np.sin(theta)*np.sin(phi)
    n_z = np.cos(theta)
    x_end = x_start + track_length*n_x
    y_end = y_start + track_length*n_y
    z_end = z_start + track_length*n_z

    # Set up Boundary conditions
    start_boundary = 50
    z_min_start = -505 - start_boundary
    z_max_start = -155 + start_boundary
    end_boundary = 50
    z_min_end = -505 - end_boundary
    z_max_end = 505 + end_boundary
    radius_IC7 = 150
    radius_DC = 90
    radius_IC19 = 260

    old_z_mask_start = np.logical_and(z_start > -505, z_start < 192)
    z_mask_start = np.logical_and(z_start > z_min_start, z_start < z_max_start)
    r_start = np.sqrt( (x_start - x_origin)**2 + (y_start - y_origin)**2 )
    z_mask_end = np.logical_and(z_end > z_min_end, z_end < z_max_end)
    r_end = np.sqrt((x_end - x_origin)**2 + (y_end - y_origin)**2)

    vertex_mask = {}
    vertex_mask["all_start"] = np.ones((len(theta)),dtype=bool)
    vertex_mask["old_start_DC"] = np.logical_and(old_z_mask_start, r_start < radius_DC)
    vertex_mask["start_DC"] = np.logical_and(z_mask_start, r_start < radius_DC)
    vertex_mask["start_IC7"] = np.logical_and(z_mask_start, r_start < radius_IC7)
    vertex_mask["start_IC19"] = np.logical_and(z_mask_start, r_start < radius_IC19)
    vertex_mask["all_end"] = np.ones((len(theta)),dtype=bool)
    vertex_mask["end_IC7"] = np.logical_and(z_mask_end, r_end < radius_IC7)
    vertex_mask["end_IC19"] = np.logical_and(z_mask_end, r_end < radius_IC19)

    return vertex_mask


def Shuffler(full_features_DC, full_features_IC, full_labels, full_reco=None, full_initial_stats=None, full_num_pulses=None,full_trig_times=None, full_weights=None, use_old_reco_flag=False):
    """Shuffle the contents of the arrays
        Receives:
        full_features_DC = fully concatenated DC array
        full_features_IC = fully concatenated IC array
        full_labels = fully concatenated labels array
        Outputs:
        shuffled_features_DC = shuffled full DC array
        shuffled_features_IC = shuffled full IC array
        shuffled_labels = shuffled full labels array
    """
    shuffled_features_DC = np.zeros_like(full_features_DC)
    shuffled_features_IC = np.zeros_like(full_features_IC)
    shuffled_labels = np.zeros_like(full_labels)
    if full_reco is not None:
        shuffled_reco = np.zeros_like(full_reco)
    else:
        shuffled_reco = None
    if full_initial_stats is not None:
        shuffled_initial_stats = np.zeros_like(full_initial_stats)
    else:
        shuffled_initial_stats = None
    if full_num_pulses is not None:
        shuffled_num_pulses = np.zeros_like(full_num_pulses)
    else:
        shuffled_num_pulses = None
    if full_trig_times is not None:
        shuffled_trig_times = np.zeros_like(full_trig_times)
    else:
        shuffled_trig_times = None
    if full_weights is not None:
        shuffled_weights = np.zeros_like(full_weights)
    else:
        shuffled_weights = None
    
    random_order = np.arange(0,full_features_DC.shape[0])
    np.random.shuffle(random_order)
    for evt_num in range(0,len(random_order)):
        shuffled_features_DC[evt_num] = full_features_DC[random_order[evt_num]]
        shuffled_features_IC[evt_num] = full_features_IC[random_order[evt_num]]
        shuffled_labels[evt_num] = full_labels[random_order[evt_num]]
        if full_reco is not None:
            shuffled_reco[evt_num] = full_reco[random_order[evt_num]]
        if full_initial_stats is not None:
            shuffled_initial_stats[evt_num] = full_initial_stats[random_order[evt_num]]
        if full_num_pulses is not None:
            shuffled_num_pulses[evt_num] = full_num_pulses[random_order[evt_num]]
        if full_trig_times is not None:
            shuffled_trig_times[evt_num] = full_trig_times[random_order[evt_num]]
        if full_weights is not None:
            shuffled_weights[evt_num] = full_weights[random_order[evt_num]]

    return shuffled_features_DC, shuffled_features_IC, shuffled_labels, shuffled_reco, shuffled_initial_stats, shuffled_num_pulses, shuffled_trig_times, shuffled_weights

def SplitTrainTest(features_DC,features_IC,labels,reco=None,weights=None,use_old_reco=False,create_validation=True,fraction_test=0.1,fraction_validate=0.2):
    """
    Splits features DC, features IC, labels, and (optionally) old reco into train, test, and validation sets
    Receives:
        features_DC = array containing input features from DC strings        
        features_IC = array containing input features from IC strings
        labels = array containing output labels
        reco = array containing old reco (PegLeg)
        use_old_reco = bool on if you will provide a reco array
        create_validation = True by default, will split training set into validation and training
        fraction_test = fraction of data to test with (default = 10%)
        fraction_validate = fraction of training data to use as validation (default = 20%)
    Outputs arrays with data split:
        X_train_DC_raw, X_train_IC_raw, Y_train_raw
        X_test_DC_raw, X_test_IC_raw, Y_test_raw
        X_validate_DC_raw, X_validate_IC_raw, Y_validate_raw
        reco_train_raw, reco_test_raw, reco_validate_raw
    """

    assert features_DC.shape[0]==features_IC.shape[0], "DC events not equal to IC events"
    assert features_DC.shape[0]==labels.shape[0], "Different number of input features than output labels"
    assert fraction_test<1.0, "Test fraction must be less than 1.0"
    assert fraction_validate<1.0, "Validate fraction must be less than 1.0"


    ### Split into training and testing set ###
    fraction_train = 1.0 - fraction_test
    num_train = int(features_DC.shape[0]*fraction_train)
    print("Testing on %.2f percent of data"%(fraction_test*100))
    if create_validation:
        num_validate = int(num_train*fraction_validate)
        print("Vadilating on %.2f percent of training data"%(fraction_validate*100))
    else:
        num_validate = 0
    print("training only on {} samples, validating on {} samples, testing on {} samples".format(num_train-num_validate, num_validate, features_DC.shape[0]-num_train))

    features_DC_train = features_DC[num_validate:num_train]
    features_IC_train = features_IC[num_validate:num_train]
    labels_train = labels[num_validate:num_train]
    if weights is not None:
        weights_train = weights[num_validate:num_train]
    if use_old_reco:
        reco_train = reco[num_validate:num_train]

    features_DC_test = features_DC[num_train:]
    features_IC_test = features_IC[num_train:]
    labels_test = labels[num_train:]
    if weights is not None:
        weights_test = weights[num_train:]
    if use_old_reco:
        reco_test = reco[num_train:]

    if create_validation:
        features_DC_validate = features_DC[:num_validate]
        features_IC_validate = features_IC[:num_validate]
        labels_validate = labels[:num_validate]
        if weights is not None:
            weights_validate = weights[:num_validate]
        if use_old_reco:
            reco_validate = reco[:num_validate]

    ### Specify type for training and testing ##
    (X_train_DC_raw, X_train_IC_raw, Y_train_raw) = (features_DC_train, features_IC_train, labels_train)
    X_train_DC_raw = X_train_DC_raw.astype("float32")
    X_train_IC_raw = X_train_IC_raw.astype("float32")
    Y_train_raw = Y_train_raw.astype("float32")
    if weights is not None:
        weights_train_raw = weights_train.astype("float32")
    if use_old_reco:
        reco_train_raw = reco_train.astype("float32")
    else:
        reco_train_raw = None

    (X_test_DC_raw, X_test_IC_raw, Y_test_raw) = (features_DC_test, features_IC_test, labels_test)
    X_test_DC_raw = X_test_DC_raw.astype("float32")
    X_test_IC_raw = X_test_IC_raw.astype("float32")
    Y_test_raw = Y_test_raw.astype("float32")
    if weights is not None:
        weights_test_raw = weights_test.astype("float32")
    if use_old_reco:
        reco_test_raw = reco_test.astype("float32")
    else:
        reco_test_raw = None

    if create_validation:
        (X_validate_DC_raw, X_validate_IC_raw,Y_validate_raw) = (features_DC_validate, features_IC_validate, labels_validate)
        X_validate_DC_raw = X_validate_DC_raw.astype("float32")
        X_validate_IC_raw = X_validate_IC_raw.astype("float32")
        Y_validate_raw = Y_validate_raw.astype("float32")
        if weights is not None:
            weights_validate_raw = weights_validate.astype("float32")
        if use_old_reco:
            reco_validate_raw = reco_validate.astype("float32")
        else:
            reco_validate_raw = None
    else:
        X_validate_DC_raw = None
        X_validate_IC_raw = None
        Y_validate_raw = None

    return X_train_DC_raw, X_train_IC_raw, Y_train_raw,  X_test_DC_raw, X_test_IC_raw, Y_test_raw, X_validate_DC_raw, X_validate_IC_raw, Y_validate_raw, reco_train_raw, reco_test_raw, reco_validate_raw, reco_train_raw, reco_test_raw, weights_validate_raw
