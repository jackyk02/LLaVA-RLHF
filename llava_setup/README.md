# Install LLaVA

## Apply the custom patch

```bash
git clone https://github.com/haotian-liu/LLaVA.git

cd LLaVA

git reset --hard 6cea223

git apply < ../fix_llava_padding.patch
```

## Install LLaVA

Please follow the instructions in the [original repository](https://github.com/haotian-liu/LLaVA/tree/6cea223532a7ab7bda8116336c59772faccdcbca#install) to install LLaVA.