parser.add_argument("--gcd_run", default=None, 
                    dest="gcd_run", help="Run Number for GCD file -- ONLY NEED FOR data") 
parser.add_argument("--gcd_year", default=None, 
                    dest="gcd_year", help="year for GCD file -- ONLY NEED FOR data")

if gcd_year is not None:
    goodrun_file = "/data/exp/IceCube/%s/filtered/level2pass2a/IC86_%s_GoodRunInfo.txt"%(gcd_year,gcd_year)
    goodinfo = np.genfromtxt(goodrun_file,skip_header=2,usecols=(0,7),dtype=str)
    run_numbers = np.array(goodinfo[:,0],dtype=int)
    paths = np.array(goodinfo[:,1])
    index_runnum = np.where(run_numbers == gcd_run)[0][0]
    gcd_path = str(paths[index_runnum][0]) + "/*_GCD.i3.zst"
    print(gcd_path)
    gcd_string = glob.glob(gcd_path)
    print(gcd_string)
    gcdfile = gcd_string #pull name from list that glob auto creates
else:
    gcdfile = None

