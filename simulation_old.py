#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import functions as fu
from scipy import special
from scipy import stats
import traceback
import sys


def run_sims_window(win,sd,periods,d,nsims,adj,mixed_norm,cluster,minsd,p,sd_arr,sds):

	a=[]
	sd_arr=sd_arr*sd
	pmsq=np.sum(p*sd_arr**2)**0.5

	avg_voladj=np.mean([np.log(voladj(d,sd_arr*(s/sd)*win**0.5,pmsq*(s/sd)*win**0.5,adj,p)) for s in sds])
	avg_voladj=np.exp(avg_voladj)
	E_abs=E_abs_func(win, sd_arr, p,cluster,mixed_norm)
	for i in range(nsims):
		r=simulation(1E-10, sd, periods, d, win,adj,mixed_norm,cluster,p,sd_arr,E_abs,pmsq,avg_voladj)
		a.append(r)	
	a=np.array(a)
	a=a+(a==0)*minsd
	a_ln=np.array(np.log(a))
	m=np.mean(a_ln,0)
	ssd=(np.sum((a_ln-m)**2,0)/(nsims-1))**0.5
	#print((np.mean(a,0),E_abs))
	m=np.concatenate((m,ssd,np.abs(m)))
	m=m.reshape((len(m),1))

	return m


def E_abs_func(win,sd_arr,p,cluster,mixed_norm):
	if mixed_norm==0:
		return sd_arr
	if cluster>1:
		if win>cluster:
			wc=win/cluster
			e1=E_abs_func2(int(wc)+1,sd_arr,p)
			e2=E_abs_func2(int(wc)+2,sd_arr,p)
			x=(wc-int(wc))
			E_abs = (1-x)*e1+x*e2
		else:
			E_abs=E_abs_func2(1,sd_arr,p)
	else:
		E_abs = E_abs_func2(win,sd_arr,p)

	return E_abs


def E_abs_func2(win,sd_arr,p):

	if type(sd_arr)==float:
		return Evar
	elif win==0: 
		return np.sum(p*sd_arr)
	elif win<=450: 
		return E_abs_calc(p, sd_arr, win)
	else:
		return np.sum(p*sd_arr**2)**0.5


def E_abs_calc(p,sd_arr,win):
	sd_arrsq=sd_arr**2
	K=len(sd_arr)
	a=np.rollaxis(np.indices([win+1]*(K-1)),0,K).reshape((win+1)**(K-1),K-1)
	a=a[np.sum(a,1)<=win]
	b=win-np.sum(a,1).reshape((len(a),1))
	a=np.concatenate((a,b),1)
	m=multinominaldist(a,p)
	ret=np.sum(a*sd_arrsq,1)**0.5
	ret2=np.sum(ret*m)/win**0.5

	return ret2

def multinominaldist(q,p):
	a=special.gammaln(np.sum(q[0])+1)
	b=np.sum(special.gammaln(q+1),1)
	c=np.prod(p**q,1)

	return np.exp(a-b)*c



def random_mixed_normal(mu,N,sd_arr,p,cluster):
	"""sd_arr should be an array of standard deviations such that np.sum(p*np.array(sd_arr))=1"""
	if cluster>1:
		N_clusters=int(N/cluster)
		r=np.random.uniform(size=N_clusters).reshape((N_clusters,1))
		r=r*np.ones((1,cluster))
	else:
		r=np.random.uniform(size=N)
	r=r.reshape((N,1))
	p1=np.cumsum(p).reshape((1,len(p)))
	p1[0,2]=1.1
	p0=np.append([[-0.1]],p1[0,:-1])
	n=np.nonzero((r>p0)*(r<=p1))
	sd_rm=sd_arr[n[1]]
	rnd=np.random.normal(mu,sd_rm,N)

	return rnd

def return_calc(p,d,N,windows,win,p0):	
	if d==0:
		p_obs=p
	else:
		p0=np.round(p0/d)*d
		rnded=np.round(p/d)
		p_obs=rnded*d
	lp_obs=np.log(p_obs)

	lp_obs=lp_obs.reshape((windows,win)).T
	ret_period=lp_obs[-1]-np.roll(lp_obs[-1],1)
	ret_period[0]=lp_obs[-1,0]-np.log(p0)

	return lp_obs,ret_period




def sim_returns(mixed_norm,mu,sd,N,d,win,cluster,p,sd_arr):
	windows=int(N/win)
	N=windows*win	

	if mixed_norm:
		rnd=random_mixed_normal(mu,N,sd_arr,p,cluster)
	else:
		rnd=np.random.normal(mu,sd,N)

	#if rho>0: not in use, clustering used instead
	#	rnd=rnd-rho*np.roll(rnd,1)

	p0=1+np.random.uniform(-d, d)#ensures it does not allways start far from a changing value
	p_cont=np.exp(np.cumsum(rnd))
	p_cont=p_cont*p0

	lp_obs,ret_period=return_calc(p_cont,d,N,windows,win,p0)
	psd=np.sum(p*sd_arr)

	return lp_obs,ret_period,sd_arr,p,windows,psd



def simulation(mu,sd,N,d,win,adj,mixed_norm,cluster,p,sd_arr,E_abs,pmsq, avg_voladj):

	lp_obs,ret_period,sd_arr,p,windows,psd=sim_returns(mixed_norm,mu,sd,N,d, win,cluster,p,sd_arr)


	rng=rng_func(lp_obs,win,adj,mixed_norm,sd,cluster,p,sd_arr,E_abs)

	msq_adj,msq_unadj,avg_abs=msq_func(win,windows,ret_period,adj,pmsq,sd_arr,d,p,psd,E_abs, avg_voladj)

	return np.array([rng,msq_adj,msq_unadj,avg_abs])


def asympt_var(d,sd):

	if d==0:
		v=1
	if d/sd<2.5:
		v=sd**2+(d**2)/6.0
	else:
		v=2*(d*sd)/((2.0*np.pi)**0.5)
	return v

def voladj(d,sd_arr,sd,adj,p):


	if adj: #for adjusting volatilities
		if type(sd_arr)==float:
			v=asympt_var(d,sd)
		else:
			v=np.sum([p[i] * asympt_var(d,sd_arr[i]) for i in range(len(sd_arr))])
		v=v**0.5
		v=v/sd
	else:
		v=1	
	return v

def msq_func(win,windows,ret,adj,pmsq,sd_arr,d,p,psd,E_abs, avg_voladj):
	if windows==1:
		return np.nan,np.nan,np.nan
	freq=len(np.nonzero(ret==0)[0])/len(ret)
	ret=ret-np.sum(ret)/(windows)
	ln_bias=(2*np.exp(special.psi((windows-1)/2))/(windows-1))**0.5
	dofa=dof_adj(windows-1,adj)


	msq=(np.sum(ret**2)/(win*(windows-1)))
	avg_abs=(np.sum(np.abs(ret))/win**0.5)/(((windows-1)*(windows))**0.5)
	vol_adj=voladj(d,sd_arr*win**0.5,pmsq*win**0.5,adj,p)


	abs_bias=ln_bias*(2/np.pi)**0.5/dofa

	avg_abs=avg_abs/abs_bias
	if adj:
		avg_abs=avg_abs/E_abs
	else:
		avg_abs=avg_abs/psd

	msq_adj=msq/(ln_bias*vol_adj*pmsq)**2
	msq_unadj=msq/(ln_bias*avg_voladj*pmsq)**2
	return msq_adj,msq_unadj,avg_abs



def dof_adj(k,adj):

	if k>250 or not adj:
		a=1
	else:
		if k<1:
			k=1
		a=2**0.5*special.gamma((k+1)/2)/(special.gamma(k/2))	
		a=a/k**0.5
	return a


def rng_func(data,win, adj,mixed_norm,sd,cluster,p,sd_arr,E_abs):
	if win==1:
		return np.nan	

	mx=np.max(data,0)
	mn=np.min(data,0)
	rng=np.mean(mx-mn)
	if not adj:
		return rng/sd

	sd_var=np.sum(p*sd_arr**2)**0.5
	ass_exp=(8/np.pi)**0.5
	adj=sd_var*ass_exp*(win-1)**0.5
	r1= rng/adj
	adj=sd*0.5*ass_exp*(win-1)**0.5
	r2= rng/adj
	a=np.exp(-0.25*(win-2))
	adj=a*0.5+(1-a)
	ret=rng/(E_abs*adj*ass_exp*(win-1)**0.5)
	return ret



