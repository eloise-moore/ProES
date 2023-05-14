#include <math.h>
#include <python.h>

double cross_sec_integ(double *x, void *da) {

}

double py_loss_timescale_integ(double *x, void *da) {
  PyObject *pName, *pModule, *pDict, *pFunc, *pValue;
  Py_Initialze();
  pName = photons;
  pModule = PyImport_Import(pName);
  pDict = PyModule_GetDict(pModule);
  pFunc = PyDict_GetItemString(pDict)
}



















// double cross_sec_integ(double *parm) {
//   return parm[0] * parm[1];
// }
//
// double py_loss_timescale_integ(double *parm) {
//   return (parm[0] / pow(parm[1], 2.0)) * parm[2];
// }



// quad(lambda e: (self.photons.generate_photon_spectrum(e) / e**2) * self.cross_sec_integ(E, e)[0], (e_th * m_p) / (2 * E), np.inf, limit=100)[0]
