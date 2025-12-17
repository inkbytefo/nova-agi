# NovaNet Mimari Dokümantasyonu

**Sürüm:** HDCT-v2 (Causal Hypergraph Architecture)
**Tarih:** 16 Aralık 2025
**Geliştirici:** inkbytefo
**Araştırma Şirketi:** Oxis Research
**Proje Lideri:** Tevfik İşkın

## 1. Genel Bakış

NovaNet, doğal dil işleme (NLP) görevleri için tasarlanmış, **Causal Deep Hypergraph Neural Network (Nedensel Derin Hipergraf Sinir Ağı)** tabanlı yeni nesil bir mimaridir. Transformer mimarilerinin $O(N^2)$ karmaşıklığını ve bellek darboğazlarını aşmak amacıyla, metinsel veriyi nedensel bir **Hipergraf (Hypergraph)** olarak modeller.

Bu yaklaşım, tokenlar arasındaki ilişkileri "Self-Attention" yerine **Causal Message Passing** (Nedensel Mesajlaşma) ile çözer. Model, eğitim sırasında gelecekteki tokenlara erişimi matematiksel olarak engelleyerek (masking) "Autoregressive" (sıralı) metin üretimini garanti altına alır.

---

## 2. Temel Bileşenler

### 2.1. Model Konfigürasyonu (TPU v3-8 / GPU Setup)

| Parametre | Değer | Açıklama |
|Lx|---|---|
| **Model Tipi** | Causal Hypergraph | Nedensel (Geçmişe bakabilen) konvolüsyon. |
| **Eğitilebilir Parametreler** | ~85M - 120M | 12 katman, 768 hidden dim. |
| **Hidden Dimension** | 768 | Düğüm (node) öznitelik vektör boyutu. |
| **Layers** | 12 | Hipergraf konvolüsyon katmanı sayısı. |
| **Vocab Size** | 5000 | "HDCT" karakter/alt-kelime seviyesi sözlük boyutu. |
| **Max Edges** | 4096+ | JIT uyumluluğu için sabitlenmiş maksimum kenar sayısı. |
| **Sequence Length** | 2048 | Maksimum bağlam uzunluğu. |

### 2.2. Nedensel Hipergraf Yapısı (Causal Hypergraph)

NovaNet, klasik Transformer Attention matrisi yerine **Ayrıştırılmış İnsidans Matrisleri ($H_{in}, H_{out}$)** kullanır. Bu yapı, bilgi akışının sadece geçmişten geleceğe ($t' \le t$) olmasını sağlar.

*   **Düğümler ($V$):** Girdi metnindeki tokenlar.
*   **Hiper-Kenarlar ($E$):** Token gruplarını bağlayan ilişkisel yapılar.

#### Kritik Yenilik: $H_{in}$ ve $H_{out}$ Ayrımı
Eski hipergraf ağlarında tek bir $H$ matrisi kullanılırken, NovaNet nedenselliği korumak için bunu ikiye ayırır:

1.  **$H_{in}$ (Gather Matrix):** Kenarların *hangi* düğümlerden bilgi toplayacağını belirler.
    *   Kural: Bir $e$ kenarı $t$ anındaki bir çıktıyı etkileyecekse, $H_{in}$ sadece $t$ ve öncesindeki ($0 \dots t$) düğümlere bağlanabilir.
2.  **$H_{out}$ (Scatter Matrix):** İşlenmiş kenar bilgisinin *nereye* dağıtılacağını belirler.
    *   Kural: Bir $e$ kenarı sadece hedef düğüm $t$'ye yazar.

Bu yapı sayesinde, $t$ anındaki düğüm güncellemesi için asla $t+1$ bilgisi kullanılamaz.

---

## 3. Çekirdek Mekanizma: Gated Causal Hypergraph Convolution

Modelin kalbi, `nova.core.ops.causal_hypergraph_conv` içerisinde tanımlanan algoritmadır.

### 3.1. Gated Mesajlaşma Süreci (Sparse Attention)

NovaNet, klasik hipergraf konvolüsyonunu "Gating" mekanizması ile güçlendirerek içerik tabanlı adreslemeyi (Content-Based Addressing) mümkün kılar.

$$
E_{src} = H_{in}^T X \quad \text{(Gather: Source Information)}
$$
$$
E_{tgt} = H_{out}^T X \quad \text{(Query: Target Information)}
$$
$$
\text{Gate} = \sigma(W_{gate}[E_{src} || E_{tgt}])
$$
$$
E_{final} = \text{ReLU}(W_{edge} E_{src}) \odot \text{Gate}
$$
$$
X_{local} = H_{out} E_{final} \quad \text{(Scatter: Edge to Node)}
$$

Bu yapı, modelin geçmişten gelen bilgiyi (Source) şimdiki durumuna (Target) göre filtrelemesini sağlar.

### 3.2. Causal Global Context (Kümülatif Toplam)
Eski "Global Edge" yaklaşımı, tüm dizinin ortalamasını aldığı için geleceği sızdırıyordu (Leakage). NovaNet bunu **Prefix Scan (Cumulative Sum)** ile değiştirmiştir:

$$
X_{global}^{(t)} = \frac{1}{t+1} \sum_{i=0}^{t} X^{(i)}
$$

Bu işlem $O(N)$ karmaşıklığındadır ve $t$ anında sadece geçmişin özetini (global context) sağlar. Gelecekten bilgi sızmaz.

$$
X_{final} = \text{LayerNorm}(X_{local} + X_{global})
$$

---

## 4. JIT ve Performans Optimizasyonu

JAX ve XLA (Accelerated Linear Algebra) üzerinde yüksek performans için "Static Shapes" (Sabit Şekiller) zorunludur.

*   **Sorun:** Metin uzunluğu değiştikçe hipergrafın kenar sayısı da değişir. Bu, JAX'in sürekli "Re-compilation" yapmasına neden olurdu.
*   **Çözüm (Static Padding):** `max_edges` parametresi ile (örn. 4096) hipergraf matrisleri ($H_{in}, H_{out}$) sabit boyutta tutulur. Kullanılmayan kenarlar "0" ile doldurulur (Zero-padding).
*   **Long-Range Edges (Dilated):** `HypergraphBuilder` artık her düğüm $t$ için $t-1, t-2, t-4, \dots$ şeklinde logaritmik uzaklıktaki düğümlere bağlanan kenarlar üretir. Bu, modelin çok uzak geçmişe $O(\log N)$ adımda erişmesini sağlar.
*   **Sonuç:** Tek bir derleme (Compilation) ile tüm eğitim süreci kesintisiz ve maksimum hızda devam eder.

---

## 5. Veri ve Eğitim Stratejisi

**HDCT (Hyper-Dimensional Compression Tokenizer):**
*   Küçük sözlük boyutu (5000), modelin morfolojik yapıları ezberlemesini değil, "öğrenmesini" sağlar.
*   Özellikle Türkçe gibi sondan eklemeli diller için karakter ve hece tabanlı hibrit bir yaklaşım sunar.

**Müfredat (Curriculum):**
*   Model önce basit n-gram yapılarını (yerel ilişkiler), ardından global bağlamı öğrenir.
*   Veri kaynakları: `ytu-ce-cosmos`, `BellaTurca`, `the-stack-smol`.

---

## 6. Teknik Avantajlar

1.  **Tam Nedensellik (Causality):** $H_{in}/H_{out}$ ayrımı ve `cumsum` global context ile %0 bilgi sızıntısı. Autoregressive üretim sorunsuz çalışır.
2.  **Lineer Karmaşıklık:** Attention $O(N^2)$ iken, NovaNet $O(N \cdot K)$ (K: kenar sayısı) karmaşıklığında çalışır. Uzun metinlerde (8k, 16k, 32k) bellek patlaması yaşamaz.
3.  **XLA/JIT Dostu:** Sabit tensör boyutları sayesinde TPU/GPU üzerinde maksimum throughput sağlar.
