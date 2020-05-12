#!/usr/bin/python
# -*- coding: UTF-8 -*-

import simulation as sim
import multi_core as mc
import os
import numpy as np
import functions as fu
import integration

def main():
	multiproc=True
		
	if multiproc:
		mproc=mc.multiprocess(os.cpu_count(),[['simulation','sim']])

	else:
		mproc=None
	
	for mixed_norm in [0,1,2]:
		for adj,d,name in  [[True,0.00316157992580004,'discrete']]:#,[False,0.00316157992580004,'unadjusted'],[False,0,'continous']]:
			for cluster in [1,960]:
				if not (mixed_norm!=1 and cluster>1) or True:
					fname="%s%s_%s" %(name,mixed_norm,cluster)
					simulation(mproc,5000,d,fname,adj,mixed_norm,cluster)
			
	
def simulation(mproc,nsims,d,name,adj,mixed_norm,cluster):
	#
	T=28800
	a=[]
	p=np.array([0.51067614,	0.017971288,	0.471352571])#probability

	if mixed_norm==1:#kurtosis= 2.5
		sd_arr=normalize(p, [1.477662785,	3.441783086,	0.389389192], False)
	elif mixed_norm==2:#kurtosis=50
		sd_arr=normalize(p, [0.946688649,	22.19991501,	0.249468506], False)
	else:
		sd_arr=1
		p=1
	mixedcovadj=integration.adjfactor(p, sd_arr)

	
		
	sd=0.00003
	windows=[1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 960, 1800, 3600, 7200, 14400, 28800]

	sds=    [0.000469375158868849]#, 3.60098707366965E-05, 2.61522137e-04,3.60098707366965E-05, 0.000158818654285202, 3.60098707366965E-05,0.00316157992580004/25,0.000469375158868849,2.93881922e-06,1.11522967e-05,  
	sds=    [3.60098707366965E-05, 	4.94210799812644E-05, 	5.79468667045995E-05, 	6.53806107798397E-05, 	
	         7.22187019570311E-05, 	7.88120376136979E-05, 	8.53842914052493E-05, 	9.21006887714667E-05, 	
	         9.90900794604503E-05, 	0.000106529896870796, 	0.000114509169050842, 	0.000123296858335587, 	
	         0.000133355271290085, 	0.000145046030063154, 	0.000158818654285202, 	0.000175783974006693, 	
	         0.000197889890781447, 	0.000230017140173149, 	0.000284348386931211, 	0.000469375158868849]	
	
	if not mproc is None:
		mproc.send_dict({'p':p,'sd_arr':sd_arr,'sds':sds})	
		

	if d==0:
		r=run_sims(mproc,sds[2],T,d,nsims,windows,adj,mixed_norm,cluster,p,sd_arr,sds,mixedcovadj)
		a.append(r)		
	else:
		for i in range(0,len(sds)):
			r=run_sims(mproc,sds[i],T,d,nsims,windows,adj,mixed_norm,cluster,p,sd_arr,sds,mixedcovadj)
			a.append(r)
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

def run_sims(mproc,sd,periods,d,nsims,windows,adj,mixed_norm,cluster,p,sd_arr,sds,mixedcovadj):
	a=[]
	r=[]
	cnt_d=dict()

	print ("Simulating daily sd=%.2f%%, d/sd=%.1f" %(sd*(periods**0.5)*100,d/sd))
	
	
	if not mproc is None:

		for i in windows:
			a.append('r%s=sim.run_sims_window(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)' %(i,i,sd,periods,d,nsims,adj,mixed_norm,cluster,'p','sd_arr','sds',mixedcovadj))	
		dres=mproc.execute(a)
		
		for i in windows:
			m=dres['r%s' %(i,)]
			r.append(m)

				
	else:
		for i in windows:
			m=sim.run_sims_window(i,sd,periods,d,nsims,adj,mixed_norm,cluster,p,sd_arr,sds,mixedcovadj)
			r.append(m)
	
	r=np.concatenate(r,1)
	sd=np.ones((len(r),1))*sd

	names=['range','msq_adj','avg_abs']
	n=len(names)
	names=names+[names[i]+' SD' for i in range(n)]
	names=names+[names[i]+' exp err' for i in range(n)]
	names=names+['inefficiency_square','inefficiency_abs', 'ESD_range', 'ESD_sq', 'ESD_abs']
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