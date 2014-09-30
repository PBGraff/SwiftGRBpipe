root = 'priorsample2_'
types = ['_train.txt','_test.txt','_blind.txt']
nCV = 10
CV = range(nCV)
headerline = '15,\n2,\n'

def reformatLine(line):
	line = line.rstrip('\n')
	if line[-1] == '0':
		line = line.rstrip('0')+'\n0,\n'
	else:
		line = line.rstrip('1')+'\n1,\n'
	return line

def rewriteFile(filename):
	fp = open(filename,'r')
	store = fp.readlines()
	fp.close()

	for n in range(len(store)):
		store[n] = reformatLine(store[n])

	fp = open(filename,'w')
	fp.write(headerline)
	fp.writelines(store)
	fp.close()

#for i in CV:
#	for t in types:
#		rewriteFile(root+'CV'+repr(i)+t)

#rewriteFile(root+'eval.txt')

rewriteFile("summary_list_Swiftlc_z360_lum5205_n0084_n1207_n2070_alpha065_beta300_Yonetoku_mod18_noevo_all.txt")
