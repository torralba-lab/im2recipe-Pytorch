import word2vec
import sys
import os

'''
Usage: python get_vocab.py /path/to/vocab.bin
'''
w2v_file = sys.argv[1]
model = word2vec.load(w2v_file)

vocab =  model.vocab

print("Writing to %s..." % os.path.join(os.path.dirname(w2v_file),'vocab.txt'))
f = open(os.path.join(os.path.dirname(w2v_file),'vocab.txt'),'w')
f.write("\n".join(vocab))
f.close()
