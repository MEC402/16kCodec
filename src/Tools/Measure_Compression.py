# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:08:33 2018

@author: IkerVazquezlopez
"""

import os
import sys


def usage():
      print ('Measure_Compression.py <original_file> <compressed_file>')
      sys.exit(2)

if not len(sys.argv[1:]) == 2:  
      usage()


size0 = os.stat(sys.argv[1]).st_size
size1 = os.stat(sys.argv[2]).st_size

print("Compresion rate of the compressed_file: " + str(size1/size0))

