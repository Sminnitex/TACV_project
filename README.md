#  TACV PROJECT 2024
Libreria torchattacks modificata con l'aggiunta di 5 file nella directory

```
./torchattacks/attacks
```
dm.py > Distortion Minimization attack, aggiunge la perturbazione minima necessaria per misclassificare le immagini da vere a false
lm.py > Loss maximization attack, simile ma ha meno hyperparameters da trovare
uav.py > Universal adversarial patch attack, attacco veloce e standardizzato per ogni immagine, prende una patch dell'1% dei pixel e aggiunge del rumore localizzato in quella singola patch
ulsa.py > Universal latent space attack, altro attacco standardizzato ma stavolta è missclassificare le immagini fake come reali, bisogna scegliere un image encoder, ho usato il seguente per i test
```
AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
```

bb.py > Black box transfer attack, il distortion minimization attack ma sta volta provato su un classificatore generico e simile a quello su cui stiamo facendo i test e poi applicato direttamente al classificatore che ci interessa

Le prove le ho fatte su un ResNET-50 pretrainato su IMAGENET1K_V1, modificato per forensic analysis come fatto nel paper degli attacchi. Partendo da un'accuracy del 95% l'attacco più efficace fin ora è il distortion minimization ma riesco a portarla solo al 57%, nel loro paper è molto più efficace
