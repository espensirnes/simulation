#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import scipy.integrate as integr
from scipy.stats import norm





	
def covprob(s0,s1):
	func       =lambda r1,x : norm.cdf((x-r1)/s0)*f(r1,s1)
	bounds     =lambda x       : [x,np.inf]
	res=integr.nquad(func,[bounds,[0,np.inf]])
	r=2*res[0]
	return r
	
def adjfactor(p,sd,adj=True):
	if type(sd)!=np.ndarray or adj==False:
		return 1
	P00=0.292893275851783*(2/np.pi)**0.5
	#P00*sd gives the actual P00 probability
	P01=covprob_w(sd[0],sd[1])	
	P02=covprob_w(sd[0],sd[2])	
	P12=covprob_w(sd[1],sd[2])	
	m=np.array([
	    [P00*sd[0],	P01,		P02		 ], 
	    [P01,		P00*sd[1],	P12		 ], 
	    [P02,		P12,		P00*sd[2] ]

	])
	r=np.sum(p.reshape((3,1))*p.reshape((1,3))*m)
	ret=r/(P00*np.sum(sd*p))#relative to expected 
	return ret


def covprob_w(s1,s0):
	if s1==0 or s0==0 :
		return 0
	a=covprob(s1,s0)	
	return a
	


	
def f(x,s):
	r=1/(s*(2*np.pi)**0.5)
	r=r*np.exp(-0.5*(x/s)**2)
	return r



