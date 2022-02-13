# BWT.py
# HW1, Computational Genomics, Spring 2022
# andrewid: shutingc

# WARNING: Do not change the file name, or the function signatures below.
# Autograder expects these names exactly.

def rle(s):
	"""Run Length Encoder
	Args: s, string to be encoded
	Returns: RLE(s)
	"""
	out_s = ''
	pre_c = ''
	rep = 1
	i = 0
	while i < len(s):
		if s[i] == pre_c:
			rep += 1
			if i == len(s)-1 or s[i+1] != s[i]:
				out_s += s[i] + str(rep)
		else:
			rep = 1
			out_s += s[i]
		pre_c = s[i]    
		i += 1
	return out_s
	raise NotImplementedError

def bwt_encode(s):
	"""Burrows-Wheeler Transform
	Args: s, string, which must not contain '{' or '}'
	Returns: BWT(s), which contains '{' and '}'
	"""
	s = '{' + s + '}'
	i = 0
	permut_list = []
	while i < len(s):
		permut_list.append(s[i:] + s[:i])
		i += 1
	permut_list.sort()
	bwt = ''.join([x[-1] for x in permut_list])
	return bwt
	raise NotImplementedError

def bwt_decode(bwt):
	"""Inverse Burrows-Wheeler Transform
	Args: bwt, BWT'ed string, which should contain '{' and '}'
	Returns: reconstructed original string s, must not contains '{' or '}'
	"""
	permut_list = list(bwt)
	permut_list.sort()
	i = 1
	while i < len(bwt):
		for j in range(len(permut_list)):
			permut_list[j] = bwt[j] + permut_list[j]
		permut_list.sort()
		i += 1
	for s_h in permut_list:
		if s_h[0] == "{":
			s = s_h.strip("{}")
	return s
	raise NotImplementedError

def test_string(s):
	compressed = rle(s)
	bwt = bwt_encode(s)
	compressed_bwt = rle(bwt)
	reconstructed = bwt_decode(bwt)
	template = "{:25} ({:3d}) {}"
	print(template.format("original", len(s), s))
	print(template.format("bwt_enc(orig)", len(bwt), bwt))
	print(template.format("bwt_dec(bwt_enc(orig))", len(reconstructed), reconstructed))
	print(template.format("rle(orig)", len(compressed), compressed))
	print(template.format("rle(bwt_enc(orig))", len(compressed_bwt), compressed_bwt))
	print()
	print()

if __name__ == "__main__":
	# Add more of your own strings to explore for question (i)
	test_strings = ["WOOOOOHOOOOHOOOO!",
					"scottytartanscottytartanscottytartanscottytartan"]
	for s in test_strings:
		test_string(s)
