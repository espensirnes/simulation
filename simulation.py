#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import functions as fu
from scipy import special
from scipy import stats
import traceback
import sys
	
	
def run_sims_window(win,sd,periods,d,nsims,adj,mixed_norm,cluster,minsd,p,sd_arr):
	
	a=[]
	sd_arr=sd_arr*sd
	E_abs=E_abs_func(win, sd_arr, p,cluster,mixed_norm)
	for i in range(nsims):
		r=simulation(1E-10, sd, periods, d, win,adj,mixed_norm,cluster,p,sd_arr,E_abs)
		a.append(r)	
	a=np.array(a)
	a=a+(a==0)*minsd
	a_ln=np.array(np.log(a))
	m=np.mean(a_ln,0)
	ssd=(np.sum((a_ln-m)**2,0)/(nsims-1))**0.5
	#print((np.mean(a,0),E_abs))
	m=np.concatenate((m,ssd,np.abs(m)))
	m=m.reshape((len(m),1))
	
	print(m[:5])
	print(m[5:])
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
	maxwin=150.0
	Evar=np.sum(p*sd_arr**2)**0.5
	if type(sd_arr)==float:
		E_abs = Evar
	elif win>15 and win<maxwin: 
		e1=E_abs_calc(p,sd_arr,15)
		e2=Evar
		x=(win-15)/(maxwin-15)
		x=x**0.5
		E_abs =(1-x)*e1+x*e2
	elif win>=maxwin:
		E_abs=Evar
	elif win==0:
		E_abs = np.sum(p*sd_arr)
	else:
		E_abs=E_abs_calc(p,sd_arr,win)
	return E_abs


def E_abs_calc(p,sd_arr,win):
	sd_arrsq=sd_arr**2
	K=len(sd_arr)
	a=np.rollaxis(np.indices([K]*win),0,win+1).reshape(K**win,win)
	return np.sum(np.prod(p[a],1)*(np.sum(sd_arrsq[a],1)/win)**0.5)	

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
	pmsq=np.sum(p*sd_arr**2)**0.5
	return lp_obs,ret_period,sd_arr,p,windows,psd,pmsq
	
	
	
def simulation(mu,sd,N,d,win,adj,mixed_norm,cluster,p,sd_arr,E_abs):
	
	lp_obs,ret_period,sd_arr,p,windows,psd,pmsq=sim_returns(mixed_norm,mu,sd,N,d, win,cluster,p,sd_arr)
	

	rng=rng_func(lp_obs,win,adj,mixed_norm,sd,cluster,p,sd_arr)

	msq_voladj_emp,msq_ln,msq_raw,avg_abs=msq_func(win,windows,ret_period,adj,pmsq,sd_arr,d,p,psd,E_abs)
	
	return np.array([rng,msq_voladj_emp,msq_ln,msq_raw,avg_abs])


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

def msq_func2(win,windows,ret,adj,sd,sd_arr,d,p,psd,E_abs):
	if windows==1:
		return np.nan,np.nan,np.nan,np.nan
	freq=len(np.nonzero(ret==0)[0])/len(ret)
	#ret=np.random.normal(0,sd*win**0.5,windows)
	ret=ret-np.sum(ret)/(windows)
	ln_bias=(2*np.exp(special.psi((windows-1)/2)))**0.5
	dofa=dof_adj2(windows-1,adj)
	
	
	msq=(np.sum(ret**2)/win)**0.5
	avg_abs=(np.sum(np.abs(ret))/win**0.5)/windows**0.5
	vol_adj=voladj2(d,sd_arr*win**0.5,sd*win**0.5,adj,p)


	abs_bias=ln_bias*(2/np.pi)**0.5/dofa
	#abs_bias=((windows-1)/windows)**0.5*(2/np.pi)**0.5
	avg_abs=avg_abs/abs_bias
	if adj:
		avg_abs=avg_abs/E_abs
	else:
		avg_abs=avg_abs/psd
	
	msq_voladj_emp=msq_empirical(msq/(windows-1)**0.5, d/win**0.5, dofa,sd)


	msq_ln=msq/(ln_bias*vol_adj*sd)
	msq_raw=msq/(sd*windows**0.5)
	return msq_voladj_emp,msq_ln,msq_raw,avg_abs


def dof_adj2(k,adj):

	if k>250 or not adj:
		a=1
	else:
		if k<1:
			k=1
		a=2**0.5*special.gamma((k+1)/2)/(special.gamma(k/2))	
		a=a/k**0.5
	return a


def msq_func(win,windows,ret,adj,sd,sd_arr,d,p,psd,E_abs):
	if windows==1:
		return np.nan,np.nan,np.nan,np.nan
	freq=len(np.nonzero(ret==0)[0])/len(ret)
	#ret=np.random.normal(0,sd*win**0.5,windows)
	ret=ret-np.sum(ret)/(windows)
	ln_bias=(2*np.exp(special.psi((windows-1)/2))/(windows-1))**0.5
	dofa=dof_adj(windows-1,adj)
	
	
	msq=(np.sum(ret**2)/(win*(windows-1)))**0.5
	avg_abs=(np.sum(np.abs(ret))/win**0.5)/(windows-1)
	vol_adj=voladj(d,sd_arr*win**0.5,sd*win**0.5,adj,p)


	abs_bias=ln_bias*(2/np.pi)**0.5/dofa
	#abs_bias=((windows-1)/windows)**0.5*(2/np.pi)**0.5
	avg_abs=avg_abs/abs_bias
	if adj:
		avg_abs=avg_abs/E_abs
	else:
		avg_abs=avg_abs/psd
	
	msq_voladj_emp=msq_empirical(msq, d/win**0.5, dofa,sd)


	msq_ln=msq/(ln_bias*vol_adj*sd)
	msq_raw=msq/sd
	return msq_voladj_emp,msq_ln,msq_raw,avg_abs



def dof_adj(k,adj):

	if k>250 or not adj:
		a=1
	else:
		if k<1:
			k=1
		a=2**0.5*special.gamma((k+1)/2)/(special.gamma(k/2))	
		a=a/k**0.5
	return a

	

def msq_empirical(msq,d,dofa,sd):
	if d==0:
		return msq/(dofa*sd)
	sd_emp_large_error=(0.5*np.pi)**0.5*msq**2/d
	sd_emp_small_error=(msq**2-(d**2/(6.0)))
	if sd_emp_small_error>0:
		sd_emp=sd_emp_small_error**0.5
	elif sd_emp_large_error>0:
		sd_emp=sd_emp_large_error
	else:
		sd_emp=d*0.01
	if d/sd_emp>2.5:
		sd_emp=sd_emp_large_error

	sd_emp=sd_emp/(dofa*sd)
	return sd_emp

	
def rng_func(data,win, adj,mixed_norm,sd,cluster,p,sd_arr):
	if win==1:
		return np.nan	

	mx=np.max(data,0)
	mn=np.min(data,0)
	rng=np.mean(mx-mn)
	if not adj:
		return rng/sd
	
	sd_ratio=sd/(np.sum(p*sd_arr**2)**0.5)
	q=1/(win/2)**0.5
	if mixed_norm==2 and cluster==1:
		rng_norm=rng_adj(win, 1, q,sd_ratio)
	else:
		if mixed_norm==2 and cluster==8:
			rng_norm1=rng_adj(win,0, q,1)
			rng_norm2=rng_adj(win, 1, q,sd_ratio)
			rng_norm=rng_norm1*q**0.6+rng_norm2*(1-q**0.6)
		else:
			rng_norm=rng_adj(win,0, q,sd_ratio)
	rng=rng/(rng_norm*sd)
	return rng


def rng_adj(win,a,q,sd_ratio):
	asy_exp=(8/np.pi)**0.5
	adj=(1-0.5*q)*asy_exp*((win-1)**0.5)*(q+(1-q)/sd_ratio**(1-a*q))

	return adj


