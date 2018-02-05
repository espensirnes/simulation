#!/usr/bin/python
# -*- coding: UTF-8 -*-

import simulation as sim
import multi_core as mc
import os
import numpy as np
import functions as fu

def main():
	multiproc=True
	
	
		
	if multiproc:
		mproc=mc.multiprocess(os.cpu_count(),[['simulation','sim']])

	else:
		mproc=None
	
	for mixed_norm in [0,1,2]:
		for adj,d,name in  [[True,0.001,'discrete'],[False,0,'unadjusted'],[True,0,'continous']]:
			for cluster in [1,8]:
				if not (mixed_norm<2 and cluster>1):
					fname="%s%s_%s" %(name,mixed_norm,cluster)
					simulation(mproc,1000,d,fname,adj,mixed_norm,cluster)
			
	
def simulation(mproc,nsims,d,name,adj,mixed_norm,cluster):
	#
	a=[]
	p=np.array([0.51067614,	0.017971288,	0.471352571])#probability
	if mixed_norm==1:#kurtosis= 2.5
		sd_arr=normalize(p, [1.477662785,	3.441783086,	0.389389192], True)
	elif mixed_norm==2:#kurtosis=50
		sd_arr=normalize(p, [0.946688649,	22.19991501,	0.249468506], True)
	else:
		sd_arr=1
		p=1
		
	if not mproc is None:
		mproc.send_dict({'p':p,'sd_arr':sd_arr})	
		
	sd=0.000003
	windows=[1,2,4,8,15,30,60,120,240,450,900,1800,3600,7200,14400,28800]
	
	if d>0:
		n=31
	else:
		n=3
		sd=sd*16**1.5		

	for i in range(2,n):
		r=run_sims(mproc,sd*i**1.5,28800,d,nsims,windows,adj,mixed_norm,cluster,sd,p,sd_arr)
		a.append(r)
		if d==0:
			break
	a=np.concatenate(a,0)
	b1=np.array(['_ID','SD']+list(windows))
	b2=np.array([nsims,sd]+len(windows)*[0])
	b1=b1.reshape((1,len(b1)))
	b2=b2.reshape((1,len(b2)))
	a=np.concatenate((b1,a,b2),0)
	fu.savevar(a,name+'.csv')
	
def normalize(p,sd,sd_norm):
	p=np.array(p)
	sd=np.array(sd)
	if sd_norm:
		d=np.sum(p*sd)
	else:
		d=np.sum(p*sd**2)**0.5
	return sd/d
	
	
def append_to_dict(d_to,d_from):
	for i in d_from:
		if not i in d_to:
			d_to[i]=[]
		d_to[i]

def run_sims(mproc,sd,periods,d,nsims,windows,adj,mixed_norm,cluster,minsd,p,sd_arr):
	a=[]
	r=[]
	cnt_d=dict()

	print ("Simulating daily sd=%.2f%%, d/sd=%.1f" %(sd*(periods**0.5)*100,d/sd))
	
	if not mproc is None:

		for i in windows:
			a.append('r%s=sim.run_sims_window(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)' %(i,i,sd,periods,d,nsims,adj,mixed_norm,cluster,minsd,'p','sd_arr'))	
		dres=mproc.execute(a)
		
		for i in windows:
			m=dres['r%s' %(i,)]
			r.append(m)

				
	else:
		for i in windows:
			m=sim.run_sims_window(i,sd,periods,d,nsims,adj,mixed_norm,cluster,minsd,p,sd_arr)
			r.append(m)
	
	r=np.concatenate(r,1)
	sd=np.ones((len(r),1))*sd

	names=['range','msq_adj','msq_unadj','avg_abs']
	n=len(names)
	names=names+[names[i]+' SD' for i in range(n)]
	names=names+[names[i]+' exp err' for i in range(n)]
	names=np.array(names).reshape((len(names),1))
	r=np.concatenate((names,sd,r),1)
	return r

def simulation_parkinson():
	N=10000
	k=10000
	rnd=np.random.uniform(-0.5,0.5,N*k)
	rnd=rnd.reshape((k,N))
	x=np.cumsum(rnd,1)
	xsq=x[:,-1]**2
	meanxsq=np.mean(xsq)
	stdx=np.mean((xsq-meanxsq)**2)**0.5
	l=np.max(x,1)-np.min(x,1)
	l_est=.393*l**2/1.09
	l_est_mean=np.mean(l_est)
	stl=np.mean((l_est-l_est_mean)**2)**0.5
	
	print (.393*np.mean(l)**2)
	print (l_est_mean)
	print (meanxsq)
	
	print(stl)
	print(stdx)
	d=0


main()
#simulation_parkinson()