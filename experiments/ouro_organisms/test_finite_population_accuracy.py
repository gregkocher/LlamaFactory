"""Artificial finite populations only; no confirmation cases or model inference."""
import math
import unittest
from scipy.stats import hypergeom
from finite_population_accuracy import finite_net_bound,population_success_bounds


class FinitePopulationAccuracyTests(unittest.TestCase):
    def test_integer_inversion_matches_brute_force(self):
        for N in range(1,14):
            for n in sorted({1,N,max(1,N//2)}):
                for x in range(n+1):
                    lo=min(M for M in range(N+1)if hypergeom.sf(x-1,N,M,n)>.025)
                    hi=max(M for M in range(N+1)if hypergeom.cdf(x,N,M,n)>.025)
                    self.assertEqual(population_success_bounds(N,n,x,.025),(lo,hi))

    def test_exact_joint_coverage_all_small_population_compositions(self):
        for N in range(1,13):
            for n in sorted({1,N,max(1,N//2)}):
                bounds={(h,b):finite_net_bound(N,n,h,b)['population_net_accuracy_difference_lower_bound']
                        for h in range(n+1)for b in range(n-h+1)}
                for H in range(N+1):
                    for B in range(N-H+1):
                        total=bad=0.;delta=(B-H)/N
                        for h in range(n+1):
                            for b in range(n-h+1):
                                c=n-h-b
                                if h>H or b>B or c>N-H-B:continue
                                prob=math.comb(H,h)*math.comb(B,b)*math.comb(N-H-B,c)/math.comb(N,n)
                                total+=prob
                                if bounds[h,b]>delta+1e-12:bad+=prob
                        self.assertAlmostEqual(total,1.,12)
                        self.assertLessEqual(bad,.05+1e-12,(N,n,H,B,bad))

    def test_census_is_exact_and_margin_strict(self):
        for h,b in [(0,0),(100,0),(0,100),(12,13)]:
            r=finite_net_bound(100,100,h,b)
            self.assertEqual(r['population_net_accuracy_difference_lower_bound'],(b-h)/100)
        self.assertFalse(finite_net_bound(100,100,5,0)['noninferiority_test_rejects_at_alpha'])
        self.assertTrue(finite_net_bound(100,100,4,0)['noninferiority_test_rejects_at_alpha'])

    def test_zero_discordance_is_not_false_certainty(self):
        r=finite_net_bound(1240,1000,0,0)
        self.assertLess(r['population_net_accuracy_difference_lower_bound'],0)
        self.assertEqual(r['beneficial_population_total_lower'],0)
        self.assertGreater(r['harmful_population_total_upper'],0)

    def test_invalid(self):
        for args in [(0,0,0,0),(10,11,0,0),(10,0,0,0),(10,5,3,3),(True,1,0,0),(10,5,-1,0),(10,5,1.5,0)]:
            with self.assertRaises(ValueError):finite_net_bound(*args)
        for kw in [{'alpha':0},{'alpha':float('nan')},{'margin':1},{'margin':-1}]:
            with self.assertRaises(ValueError):finite_net_bound(10,5,0,0,**kw)

if __name__=='__main__':unittest.main()
