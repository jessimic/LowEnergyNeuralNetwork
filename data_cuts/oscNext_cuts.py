#!/usr/bin/env python

'''
Collecting together the cuts for each processing stage
Also addign a top-level mdule to apply the cuts easily (e.g. when plotting)

Tom Stuttard
'''


def L2_DeepCore_Cut(frame) :
    '''
    Module that filters out frames that did not pass L2
    This is really a cut on the DeepCoreFilter
    '''

    #TODO Use common function with L3

    # Define filter year
    years = [13,12] #TODO 13 only

    # Get the filter mask
    #TODO How shoud we really handle the FilterMask vs QFilterMask thing?
    filter_mask = None
    if frame.Has("FilterMask") :
        filter_mask = frame["FilterMask"]
    elif frame.Has("QFilterMask") :
        filter_mask = frame["QFilterMask"]

    # Only proceed if found filters
    if filter_mask is not None :

        # Find the filter result
        # Year of filtrer is different in old data
        filter_obj = None
        for year in years :
            dc_filter_key = "DeepCoreFilter_%i" % year
            if filter_mask.has_key(dc_filter_key) :
                filter_obj = filter_mask.get(dc_filter_key)
                break

        # Check found the filter
        assert filter_obj is not None, "Could not find the DeepCore filter result"

        # Check if filter passed
        if filter_obj.condition_passed :
            return True

    return False



def is_processing_level(processing_level, ref_processing_level) :
    '''
    Helper function to support various formats for defining the processing level
    '''
    return processing_level in [ ref_processing_level, str(ref_processing_level), "level%i"%ref_processing_level, "L%i"%ref_processing_level ]


def get_cut_bool_key(processing_level) :
    '''
    Get the frame object key for the cut boolean for this processing level
    '''

    from icecube.oscNext.selection.globals import L3_CUT_BOOL_KEY, L4_CUT_BOOL_KEY, L5_CUT_BOOL_KEY, L6_CUT_BOOL_KEY, L7_CUT_BOOL_KEY

    if is_processing_level(processing_level, 3) :
        return L3_CUT_BOOL_KEY
    elif is_processing_level(processing_level, 4) :
        return L4_CUT_BOOL_KEY
    elif is_processing_level(processing_level, 5) :
        return L5_CUT_BOOL_KEY
    elif is_processing_level(processing_level, 6) :
        return L6_CUT_BOOL_KEY
    elif is_processing_level(processing_level, 7) :
        return L7_CUT_BOOL_KEY
    else :
        raise Exception("Could not get cut bool for processing stage '%s'" % processing_stage)


def oscNext_cut(frame, processing_level) :
    '''
    Module that filters out frames that did not pass the specified cutting level
    Mostly this is just cutting on a pre-defined bool
    Adding as a stand-alone function here as a user might like to call this from a plotting script
    '''

    # For L2, no oscNext-specific cut, but do cut on the DeepCore filter
    if is_processing_level(processing_level, 2) :
        return L2_DeepCore_Cut(frame)

    # For oscNext processing stages, just use the pre-computed bool
    else :
        cut_bool_key = get_cut_bool_key(processing_level)
        return frame[cut_bool_key].value

