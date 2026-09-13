"""CPU exact-outcome coverage checks; no model/evaluation data required."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from paired_net_accuracy import paired_net_bounds, outcome_bounds, outcome_probabilities, exact_power, cp_bounds


class PairedNetAccuracyTests(unittest.TestCase):
    def test_zero_discordance_does_not_collapse(self):
        for n in [1, 10, 100, 500]:
            r=paired_net_bounds(n,0,0)
            upper=-math.expm1(math.log(.05/4)/n)
            self.assertAlmostEqual(r['one_sided_lower_bound'],-upper,12)
            self.assertAlmostEqual(r['one_sided_upper_bound'],upper,12)
            self.assertIsNone(r['conditional_gain_fraction_observed'])
            self.assertEqual(r['conditional_gain_probability_interval'],[0.,1.])
        self.assertEqual(cp_bounds(0,0,.01),(0.,1.))

    def test_all_gain_all_harm_and_arm_symmetry(self):
        for n,h,b in [(1,1,0),(1,0,1),(30,30,0),(30,0,30),(100,12,13),(100,10,20)]:
            r=paired_net_bounds(n,h,b);s=paired_net_bounds(n,b,h)
            self.assertAlmostEqual(r['one_sided_lower_bound'],-s['one_sided_upper_bound'],12)
            self.assertAlmostEqual(r['two_sided_interval'][0],-s['two_sided_interval'][1],12)
            lo,hi=r['two_sided_interval'];self.assertLessEqual(-1,lo);self.assertLessEqual(lo,hi);self.assertLessEqual(hi,1)
            self.assertLessEqual(lo,r['net_accuracy_difference']);self.assertLessEqual(r['net_accuracy_difference'],hi)

    def test_vectorized_bounds_equal_scalar_every_small_sample_outcome(self):
        n=12;ks,bs,lo,hi,lt,ut=outcome_bounds(n)
        for i,(k,b)in enumerate(zip(ks,bs)):
            r=paired_net_bounds(n,int(k-b),int(b))
            np.testing.assert_allclose([lo[i],hi[i],lt[i],ut[i]],
                                      [r['one_sided_lower_bound'],r['one_sided_upper_bound'],*r['two_sided_interval']],atol=1e-13,rtol=0)

    def test_exact_coverage_enumeration_small_n_entire_grid(self):
        # Includes p=0, p=1, zero gains/harms, both corners, interior and small N.
        for n in [1,2,5,10,20]:
            for alpha in [.05,.2]:
                ks,bs,lo,hi,lt,ut=outcome_bounds(n,alpha=alpha)
                for ih in range(11):
                    for ib in range(11-ih):
                        h,b=ih/10,ib/10;delta=b-h
                        probs=outcome_probabilities(n,ks,bs,h,b)
                        self.assertAlmostEqual(float(probs.sum()),1.,12)
                        for bad in [lo>delta+1e-12,hi<delta-1e-12,(lt>delta+1e-12)|(ut<delta-1e-12)]:
                            self.assertLessEqual(float(probs[bad].sum()),alpha+1e-11,(n,alpha,h,b))

    def test_noninferiority_null_boundary_type_one_error(self):
        n=30;alpha=.05;margin=.05;ks,bs,lower,*_=outcome_bounds(n,alpha=alpha)
        for h in [.05,.1,.2,.3,.4,.5]:
            b=h-margin
            probs=outcome_probabilities(n,ks,bs,h,b)
            self.assertLessEqual(float(probs[lower > -margin].sum()),alpha+1e-11)

    def test_net_mean_distinct_from_harm_rate_and_point_screen(self):
        r=paired_net_bounds(100,10,20)
        self.assertTrue(r['noninferiority_test_rejects_at_alpha'])
        self.assertGreater(r['harmful']/r['n'],.05)
        r=paired_net_bounds(100,12,13)
        self.assertGreater(r['net_accuracy_difference'],0)
        self.assertFalse(r['noninferiority_test_rejects_at_alpha'])

    def test_power_hand_calculation_and_mass(self):
        # For n=1 and margin=.99, all-concordant and one-gain reject, one-harm does not.
        r=exact_power(1,.2,.3,margin=.99)
        self.assertAlmostEqual(r['probability_of_noninferiority_rejection'],.8,12)
        self.assertAlmostEqual(r['probability_mass_check'],1.,12)

    def test_invalid_inputs(self):
        for args in [(0,0,0),(10,8,3),(10,-1,0),(True,0,0),(10,1.2,0)]:
            with self.assertRaises(ValueError):paired_net_bounds(*args)
        for kwargs in [{'alpha':0},{'alpha':1},{'alpha':float('nan')},{'margin':-1},{'margin':1}]:
            with self.assertRaises(ValueError):paired_net_bounds(10,1,1,**kwargs)
        with self.assertRaises(ValueError):exact_power(10,.8,.3)

    def test_cli_immutable_output_and_no_qualification_override(self):
        with tempfile.TemporaryDirectory()as tmp:
            out=Path(tmp)/'report';script=Path(__file__).with_name('paired_net_accuracy.py')
            cmd=[sys.executable,str(script),'counts','--n','100','--harmful','12','--beneficial','13',
                 '--label','hypothetical_test','--output',str(out)]
            subprocess.run(cmd,check=True,capture_output=True,text=True)
            raw=(out/'report.json').read_bytes();x=json.loads(raw)
            self.assertFalse(x['result']['qualification_criteria_changed'])
            self.assertIn('SUPPLEMENTAL',x['scope'])
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertEqual(hashlib.sha256(raw).digest(),hashlib.sha256((out/'report.json').read_bytes()).digest())


if __name__=='__main__':unittest.main()
