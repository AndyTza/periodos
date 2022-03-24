# Periodos (περίοδος) 
Analysis on period finding in the era of the Vera C. Rubin Observatory.



### Running `lsp_summarize`
[WIP]
`lsp_summarize.py` complies all four major types of periodograms that are supported on `gatspy`: multiband and single (fast and general) Lomb-Scargle periodogram. 

Flags: 
`-N`: Number of injected light curves
`-class`: Class of light curves (currently supports rrl, eb, agn, tde)
`-kmax`: Maximum number of Fourier components to evaluate the LSP
`-fmin`: Minimum search period 
`-fmax`: Maximum search period (if fmax>dt; fmax will be set at fmax=max(dt)-5)
`-dur`: Maximum baseline of data will be restricted to `dur` days
`-dets`: Detection type (default is 'all' that include both detections & non-detections)

```python
./lsp_summarize.py -N 100 -class rrl -kmax 7 -fmin 0.1 -fmax 150 -dur 365 -dets all
```