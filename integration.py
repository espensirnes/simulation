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
	x00=0.233695
	x01=covprob_w(sd[0],sd[1])	
	x02=covprob_w(sd[0],sd[2])	
	x12=covprob_w(sd[1],sd[2])	
	m=np.array([
	    [x00*sd[0],	x01,		x02		 ], 
	    [x01,		x00*sd[1],	x12		 ], 
	    [x02,		x12,		x00*sd[2] ]

	])
	r=np.sum(p.reshape((3,1))*p.reshape((1,3))*m)
	r=r/(np.sum(sd*p)*(2/np.pi)**0.5)
	return r*(2/np.pi)**0.5/x00


def covprob_w(s1,s0):
	if s1==0 or s0==0 :
		return 0
	a=covprob(s1,s0)	
	return a
	


	
def f(x,s):
	r=1/(s*(2*np.pi)**0.5)
	r=r*np.exp(-0.5*(x/s)**2)
	return r



