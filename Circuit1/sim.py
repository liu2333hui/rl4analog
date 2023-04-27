import os

def simulate(params, scs = "test1",verbose = False): 
    # 先写params参文件
    with open("params.scs","w") as f:
        f.write("parameters ")
        for k,i in params.items():
            f.write(k+"="+i)
            f.write(" ") 

    #a = os.popen(
    cmd = "spectre "+scs+".scs -format psfascii -info -note -debug -inter -warn -log" 
    if(verbose):
        os.system(cmd)
    else:
        os.system(cmd + " > /dev/null") 


    DC_gain = 0
    DB3_gain = 0
    Freq = 0
    DB3_Freq = 0 
    with open(scs+".raw/ac.ac") as g:
        lines = g.readlines()
        idx = 0
        state = 0
        while idx < len(lines):
            l = lines[idx].strip()		
            if(state == 0): 
                if("VALUE" == l):
                    state = 1
            elif(state == 1):
                
                #print (l)
                if("END" == l):
                    state = 2
                    continue
                #freq 
                freq = float(l.split()[1])
                idx+=1
                #input
                l = lines[idx].strip()
                输入 = float(l.split()[1][1:])
                idx+=1
                #output
                l = lines[idx].strip()
                输出 = float(l.split()[1][1:])
               
                增益 = 输出 / 输入 

                if(DC_gain == 0):
                    DC_gain = 增益
                    Freq = freq
                else:
                    if(增益 < DC_gain/1.41 ):
                        DB3_gain = 增益
                        DB3_Freq = freq
                        state = 2
            elif(state == 2):
                break 

            idx += 1	
    return DC_gain, Freq, DB3_gain, DB3_Freq


if __name__ == "__main__":
    params = {
	"K": "1.35",
	"W6":"1125n",
	"W11":"1000n",
	"VN2":"900m",
	"S": "2",
	"VN": "900m",
	"VP": "300m",
	"W8": "4u",
	"F": "1k",
	"W12":"4.5u",
	"W2":"600n",
	"W1":"6u",
	"A":"0.01",
	"L":"245n",
	"W10":"W8",
	"W7":"W6"
    }
    params_2 = {
	"K": "1.35",
	"W6":"2.25u",
	"W11":"2u",
	"VN2":"900m",
	"S": "2",
	"VN": "900m",
	"VP": "300m",
	"W8": "8u",
	"F": "1k",
	"W12":"9u",
	"W2":"1.2u",
	"W1":"9u",
	"A":"0.01",
	"L":"450n",
	"W10":"W8",
	"W7":"W6"
    }
    print(simulate(params_2,scs="test",verbose=True))
