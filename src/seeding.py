"""
Ortak tohumlama (seed) yardımcısı.

seed_everything() eskiden src/comparison.py içindeydi; src/experiment.py de
kullandığı için buraya taşındı (kopyalanmadı). comparison.py bu fonksiyonu
buradan içe aktarır, davranışı birebir aynıdır.

torch bu modülde İÇE AKTARILMAZ: yalnızca zaten yüklenmişse (ör. REINFORCE
modülleri import edildiyse) tohumlanır. Böylece random / Q-Learning işleri
PyTorch kurulu olmayan makinede de çalışır. torch'u kullanan her kod yolu
(reinforce_agent.py) torch'u modül seviyesinde import ettiği için, torch'a
bağlı bir sonuç üretilecekse torch bu fonksiyon çağrılmadan önce yüklenmiş olur.
"""

from __future__ import annotations

import random
import sys

import numpy as np


def seed_everything(seed: int) -> None:
    """Tüm GLOBAL rastgelelik kaynaklarını sabitler — tekrarlanabilirlik için.

    Profiller ve ajan parametre varyasyonu zaten seed'li `np.random.default_rng`
    kullanıyordu; Baseline bu yüzden her çalıştırmada birebir aynı çıkıyordu.
    Ancak öğrenen ajanlar seed'lenmemiş GLOBAL RNG'lere bağlıydı:

        ql_agent.py       : np.random.random() / np.random.randint()  → ε-greedy keşif
        reinforce_agent.py: torch.randn_like()                        → Gauss gürültüsü
        PolicyNet         : torch varsayılan ağırlık başlangıcı

    Bu yüzden aynı komut aynı kodla farklı QL/NN sonuçları veriyordu (ör. QL satıcı
    geliri bir çalıştırmada +%19.4, diğerinde −%9.1). Burada hepsi sabitleniyor.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
