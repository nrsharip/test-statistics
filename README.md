# Test Statistics

(The test numbering matches the numbering in the book ["100 Statistical Tests 3rd Edition"](https://books.google.com/books?id=c16MhjA4pHgC) by [Gopal K Kanji](https://en.wikipedia.org/wiki/Gopal_Kanji))

* [01. Z-test for a population mean (variance known)](#z-test-for-a-population-mean-variance-known)
* [03. Z-test for two population means (variances known and unequal)](#z-test-for-two-population-means-variances-known-and-unequal)
* [04. Z-test for a proportion (binomial distribution)](#z-test-for-a-proportion-binomial-distribution)
* [05. Z-test for the equality of two proportions (binomial distribution)](#z-test-for-the-equality-of-two-proportions-binomial-distribution)
* [07. t-test for a population mean (variance unknown)](#t-test-for-a-population-mean-variance-unknown)
* [08. t-test for two population means (variances unknown but equal)](#t-test-for-two-population-means-variances-unknown-but-equal)
* [09. t-test for two population means (variances unknown and unequal)](#t-test-for-two-population-means-variances-unknown-and-unequal)
* [15. χ2-test for a population variance](#χ2-test-for-a-population-variance)
* [16. F-test for two population variances (variance ratio test)](#f-test-for-two-population-variances-variance-ratio-test)
* [22. F-test for K population means (analysis of variance)](#f-test-for-k-population-means-analysis-of-variance)

## Z-test for a population mean (variance known)

* Sample Size n=5
  ![](docs/01-n-005.gif)
* Sample Size n=20
  ![](docs/01-n-020.gif)
* Sample Size n=100
  ![](docs/01-n-100.gif)

## Z-test for two population means (variances known and unequal)

* μ1 = μ2 = 10, σ1 = σ2 = 4
  * n1=10, n2=5
    ![](docs/03-M-S-n-10-4-010-10-4-005.gif)
  * n1=20, n2=40
    ![](docs/03-M-S-n-10-4-020-10-4-040.gif)
  * n1=200, n2=100
    ![](docs/03-M-S-n-10-4-200-10-4-100.gif)
* μ1 = 15, μ2 = 10, σ1 = 2, σ2 = 4
  * n1=10, n2=5
    ![](docs/03-M-S-n-15-2-010-10-4-005.gif)
  * n1=20, n2=40
    ![](docs/03-M-S-n-15-2-020-10-4-040.gif)
  * n1=200, n2=100
    ![](docs/03-M-S-n-15-2-200-10-4-100.gif)

## Z-test for a proportion (binomial distribution)

* Sample Size n=40
  ![](docs/04-n-0040.gif)
* Sample Size n=200
  ![](docs/04-n-0200.gif)
* Sample Size n=1000
  ![](docs/04-n-1000.gif)

## Z-test for the equality of two proportions (binomial distribution)

* n1=40, n2=100
  ![](docs/05-P-n-07-0040-05-0100.gif)
* n1=200, n2=400
  ![](docs/05-P-n-07-0200-05-0400.gif)
* n1=1000, n2=500
  ![](docs/05-P-n-07-1000-05-0500.gif)

## t-test for a population mean (variance unknown)

* Sample Size n=5
  ![](docs/07-n-005.gif)
* Sample Size n=20
  ![](docs/07-n-020.gif)
* Sample Size n=100
  ![](docs/07-n-100.gif)

## t-test for two population means (variances unknown but equal)

* μ1 = μ2 = 10, σ1 = σ2 = 3
  * n1=10, n2=5
    ![](docs/08-M-S-n-10-3-010-10-3-005.gif)
  * n1=20, n2=40
    ![](docs/08-M-S-n-10-3-020-10-3-040.gif)
  * n1=200, n2=100
    ![](docs/08-M-S-n-10-3-200-10-3-100.gif)
* μ1 = 12, μ2 = 10, σ1 = σ2 = 3
  * n1=10, n2=5
    ![](docs/08-M-S-n-12-3-010-10-3-005.gif)
  * n1=20, n2=40
    ![](docs/08-M-S-n-12-3-020-10-3-040.gif)
  * n1=200, n2=100
    ![](docs/08-M-S-n-12-3-200-10-3-100.gif)

## t-test for two population means (variances unknown and unequal)

* μ1 = μ2 = 10, σ1 = 2, σ2 = 4
  * n1=10, n2=5
    ![](docs/09-M-S-n-10-2-010-10-4-005.gif)
  * n1=20, n2=40
    ![](docs/09-M-S-n-10-2-020-10-4-040.gif)
  * n1=200, n2=100
    ![](docs/09-M-S-n-10-2-200-10-4-100.gif)
* μ1 = 20, μ2 = 15, σ1 = 2, σ2 = 4
  * n1=10, n2=5
    ![](docs/09-M-S-n-20-2-010-15-4-005.gif)
  * n1=20, n2=40
    ![](docs/09-M-S-n-20-2-020-15-4-040.gif)
  * n1=200, n2=100
    ![](docs/09-M-S-n-20-2-200-15-4-100.gif)

## χ2-test for a population variance

* Sample Size n=5
  ![](docs/15-n-005.gif)
* Sample Size n=20
  ![](docs/15-n-020.gif)
* Sample Size n=100
  ![](docs/15-n-100.gif)

## F-test for two population variances (variance ratio test)

* μ1 = 10, μ2 = 15, σ1 = σ2 = 2
  * n1=10, n2=5
    ![](docs/16-M-S-n-10-2-010-15-2-005.gif)
  * n1=20, n2=40
    ![](docs/16-M-S-n-10-2-020-15-2-040.gif)
  * n1=200, n2=100
    ![](docs/16-M-S-n-10-2-200-15-2-100.gif)

## F-test for K population means (analysis of variance)

* μ1 = μ2 = μ3 = μ4 = 10
  * n1=5, n2=10, n3=15, n4=20
    ![](docs/22-M-n-10-005-10-010-10-015-10-020.gif)
  * n1=50, n2=100, n3=150, n4=200
    ![](docs/22-M-n-10-050-10-100-10-150-10-200.gif)

* μ1 = 25, μ2 = 20, μ3 = 15, μ4 = 10
  * n1=5, n2=10, n3=15, n4=20
    ![](docs/22-M-n-25-005-20-010-15-015-10-020.gif)

# License

This project is available under the [MIT license](LICENSE) © Nail Sharipov