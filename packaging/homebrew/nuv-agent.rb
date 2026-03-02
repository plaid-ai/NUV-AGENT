class NuvAgent < Formula
  include Language::Python::Virtualenv

  desc "Nuvion on-device agent"
  homepage "https://github.com/plaid-ai/NUV-agent"
  url "__URL__"
  sha256 "__SHA256__"
  version "0.1.28"
  license "Proprietary"

  depends_on "python@3.14"
  depends_on "gstreamer"
  depends_on "gst-plugins-base"
  depends_on "gst-plugins-good"
  depends_on "gst-plugins-bad"
  depends_on "gst-plugins-ugly"
  depends_on "gst-libav"
  depends_on "ffmpeg"
  depends_on "gobject-introspection"
  depends_on "pygobject3"

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/25/7a/1fee4329abc705a469a4afe6e69b1ef7e915117747886327104a8493a955/numpy-2.4.2-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "d1240d50adff70c2a88217698ca844723068533f3f5c5fa6ee2e3220e3bdb000"
  end

  resource "python-dotenv" do
    url "https://files.pythonhosted.org/packages/14/1b/a298b06749107c305e1fe0f814c6c74aea7b2f1e10989cb30f544a1b3253/python_dotenv-1.2.1-py3-none-any.whl"
    sha256 "b81ee9561e9ca4004139c6cbba3a238c32b03e4894671e181b671e8cb8425d61"
  end

  resource "qrcode" do
    url "https://files.pythonhosted.org/packages/dd/b8/d2d6d731733f51684bbf76bf34dab3b70a9148e8f2cef2bb544fccec681a/qrcode-8.2-py3-none-any.whl"
    sha256 "16e64e0716c14960108e85d853062c9e8bba5ca8252c0b4d0231b9df4060ff4f"
  end

  resource "websockets" do
    url "https://files.pythonhosted.org/packages/bb/f4/d3c9220d818ee955ae390cf319a7c7a467beceb24f05ee7aaaa2414345ba/websockets-16.0-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "fd3cb4adb94a2a6e2b7c0d8d05cb94e6f1c81a0cf9dc2694fb65c7e8d94c42e4"
  end

  resource "stomper" do
    url "https://files.pythonhosted.org/packages/1c/f0/e924d61cf68e566f3ed5f83471230453e4e3535e330e65ed4f39cf9dfc5d/stomper-0.4.3-py3-none-any.whl"
    sha256 "e0d157db9bc1d5e2d0356bcd322fa9010d9d02cf403ca97618e8454d2697f0a9"
  end

  resource "aiohttp" do
    url "https://files.pythonhosted.org/packages/e8/0b/b97660c5fd05d3495b4eb27f2d0ef18dc1dc4eff7511a9bf371397ff0264/aiohttp-3.13.3-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "c685f2d80bb67ca8c3837823ad76196b3694b0159d232206d1e461d3d434666f"
  end

  resource "aiohappyeyeballs" do
    url "https://files.pythonhosted.org/packages/0f/15/5bf3b99495fb160b63f95972b81750f18f7f4e02ad051373b669d17d44f2/aiohappyeyeballs-2.6.1-py3-none-any.whl"
    sha256 "f349ba8f4b75cb25c99c5c2d84e997e485204d2902a9597802b0371f09331fb8"
  end

  resource "aiosignal" do
    url "https://files.pythonhosted.org/packages/fb/76/641ae371508676492379f16e2fa48f4e2c11741bd63c48be4b12a6b09cba/aiosignal-1.4.0-py3-none-any.whl"
    sha256 "053243f8b92b990551949e63930a839ff0cf0b0ebbe0597b0f3fb19e1a0fe82e"
  end

  resource "attrs" do
    url "https://files.pythonhosted.org/packages/3a/2a/7cc015f5b9f5db42b7d48157e23356022889fc354a2813c15934b7cb5c0e/attrs-25.4.0-py3-none-any.whl"
    sha256 "adcf7e2a1fb3b36ac48d97835bb6d8ade15b8dcce26aba8bf1d14847b57a3373"
  end

  resource "frozenlist" do
    url "https://files.pythonhosted.org/packages/a1/93/72b1736d68f03fda5fdf0f2180fb6caaae3894f1b854d006ac61ecc727ee/frozenlist-1.8.0-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "4970ece02dbc8c3a92fcc5228e36a3e933a01a999f7094ff7c23fbd2beeaa67c"
  end

  resource "multidict" do
    url "https://files.pythonhosted.org/packages/f1/4f/733c48f270565d78b4544f2baddc2fb2a245e5a8640254b12c36ac7ac68e/multidict-6.7.1-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "0e161ddf326db5577c3a4cc2d8648f81456e8a20d40415541587a71620d7a7d1"
  end

  resource "propcache" do
    url "https://files.pythonhosted.org/packages/b2/fa/89a8ef0468d5833a23fff277b143d0573897cf75bd56670a6d28126c7d68/propcache-0.4.1-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "9f302f4783709a78240ebc311b793f123328716a60911d667e0c036bc5dcbded"
  end

  resource "yarl" do
    url "https://files.pythonhosted.org/packages/11/63/ff458113c5c2dac9a9719ac68ee7c947cb621432bcf28c9972b1c0e83938/yarl-1.22.0-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "594fcab1032e2d2cc3321bb2e51271e7cd2b516c7d9aee780ece81b07ff8244b"
  end

  resource "idna" do
    url "https://files.pythonhosted.org/packages/0e/61/66938bbb5fc52dbdf84594873d5b51fb1f7c7794e9c0f5bd885f30bc507b/idna-3.11-py3-none-any.whl"
    sha256 "771a87f49d9defaf64091e6e6fe9c18d4833f140bd19464795bc32d966ca37ea"
  end

  resource "future" do
    url "https://files.pythonhosted.org/packages/da/71/ae30dadffc90b9006d77af76b393cb9dfbfc9629f339fc1574a1c52e6806/future-1.0.0-py3-none-any.whl"
    sha256 "929292d34f5872e70396626ef385ec22355a1fae8ad29e1a734c3e43f9fbc216"
  end

  resource "typing-extensions" do
    url "https://files.pythonhosted.org/packages/18/67/36e9267722cc04a6b9f15c7f3441c2363321a3ea07da7ae0c0707beb2a9c/typing_extensions-4.15.0-py3-none-any.whl"
    sha256 "f0fa19c6845758ab08074a0cfa8b7aecb71c999ca73d62883bc25cc018c4e548"
  end

  resource "urllib3" do
    url "https://files.pythonhosted.org/packages/39/08/aaaad47bc4e9dc8c725e68f9d04865dbcb2052843ff09c97b08904852d84/urllib3-2.6.3-py3-none-any.whl"
    sha256 "bf272323e553dfb2e87d9bfd225ca7b0f467b919d7bbd355436d3fd37cb0acd4"
  end

  resource "anyio" do
    url "https://files.pythonhosted.org/packages/38/0e/27be9fdef66e72d64c0cdc3cc2823101b80585f8119b5c112c2e8f5f7dab/anyio-4.12.1-py3-none-any.whl"
    sha256 "d405828884fc140aa80a3c667b8beed277f1dfedec42ba031bd6ac3db606ab6c"
  end

  resource "certifi" do
    url "https://files.pythonhosted.org/packages/e6/ad/3cc14f097111b4de0040c83a525973216457bbeeb63739ef1ed275c1c021/certifi-2026.1.4-py3-none-any.whl"
    sha256 "9943707519e4add1115f44c2bc244f782c0249876bf51b6599fee1ffbedd685c"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/98/78/01c019cdb5d6498122777c1a43056ebb3ebfeef2076d9d026bfe15583b2b/click-8.3.1-py3-none-any.whl"
    sha256 "981153a64e25f12d547d3426c367a4857371575ee7ad18df2a6183ab0545b2a6"
  end

  resource "charset_normalizer" do
    url "https://files.pythonhosted.org/packages/2a/35/7051599bd493e62411d6ede36fd5af83a38f37c4767b92884df7301db25d/charset_normalizer-3.4.4-cp314-cp314-macosx_10_13_universal2.whl"
    sha256 "da3326d9e65ef63a817ecbcc0df6e94463713b754fe293eaa03da99befb9a5bd"
  end

  resource "filelock" do
    url "https://files.pythonhosted.org/packages/b5/36/7fb70f04bf00bc646cd5bb45aa9eddb15e19437a28b8fb2b4a5249fac770/filelock-3.20.3-py3-none-any.whl"
    sha256 "4b0dda527ee31078689fc205ec4f1c1bf7d56cf88b6dc9426c4f230e46c2dce1"
  end

  resource "fsspec" do
    url "https://files.pythonhosted.org/packages/01/c9/97cc5aae1648dcb851958a3ddf73ccd7dbe5650d95203ecb4d7720b4cdbf/fsspec-2026.1.0-py3-none-any.whl"
    sha256 "cb76aa913c2285a3b49bdd5fc55b1d7c708d7208126b60f2eb8194fe1b4cbdcc"
  end

  resource "h11" do
    url "https://files.pythonhosted.org/packages/04/4b/29cac41a4d98d144bf5f6d33995617b185d14b22401f75ca86f384e87ff1/h11-0.16.0-py3-none-any.whl"
    sha256 "63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86"
  end

  resource "brotli" do
    url "https://files.pythonhosted.org/packages/17/e1/298c2ddf786bb7347a1cd71d63a347a79e5712a7c0cba9e3c3458ebd976f/brotli-1.2.0-cp314-cp314-macosx_10_15_universal2.whl"
    sha256 "6c12dad5cd04530323e723787ff762bac749a7b256a5bece32b2243dd5c27b21"
  end

  resource "gevent" do
    url "https://files.pythonhosted.org/packages/15/1a/948f8167b2cdce573cf01cec07afc64d0456dc134b07900b26ac7018b37e/gevent-25.9.1-cp314-cp314-macosx_11_0_universal2.whl"
    sha256 "1a3fe4ea1c312dbf6b375b416925036fe79a40054e6bf6248ee46526ea628be1"
  end

  resource "geventhttpclient" do
    url "https://files.pythonhosted.org/packages/c6/bd/78949448d1a6874c2b183bbf60ed570527b30cdfbbe0fb8252ddc441fa41/geventhttpclient-2.3.8-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "39da3e44a92ee896bd78414023f547c3481861e90181d18e8a85ca030735b0d3"
  end

  resource "greenlet" do
    url "https://files.pythonhosted.org/packages/3f/ae/8bffcbd373b57a5992cd077cbe8858fff39110480a9d50697091faea6f39/greenlet-3.3.2-cp314-cp314-macosx_11_0_universal2.whl"
    sha256 "8d1658d7291f9859beed69a776c10822a0a799bc4bfe1bd4272bb60e62507dab"
  end

  resource "hf-xet" do
    url "https://files.pythonhosted.org/packages/7f/8c/c5becfa53234299bc2210ba314eaaae36c2875e0045809b82e40a9544f0c/hf_xet-1.2.0-cp37-abi3-macosx_11_0_arm64.whl"
    sha256 "27df617a076420d8845bea087f59303da8be17ed7ec0cd7ee3b9b9f579dff0e4"
  end

  resource "httpcore" do
    url "https://files.pythonhosted.org/packages/7e/f5/f66802a942d491edb555dd61e3a9961140fd64c90bce1eafd741609d334d/httpcore-1.0.9-py3-none-any.whl"
    sha256 "2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55"
  end

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/2a/39/e50c7c3a983047577ee07d2a9e53faf5a69493943ec3f6a384bdc792deb2/httpx-0.28.1-py3-none-any.whl"
    sha256 "d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad"
  end

  resource "huggingface_hub" do
    url "https://files.pythonhosted.org/packages/a8/af/48ac8483240de756d2438c380746e7130d1c6f75802ef22f3c6d49982787/huggingface_hub-0.36.2-py3-none-any.whl"
    sha256 "48f0c8eac16145dfce371e9d2d7772854a4f591bcb56c9cf548accf531d54270"
  end

  resource "jinja2" do
    url "https://files.pythonhosted.org/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl"
    sha256 "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67"
  end

  resource "markupsafe" do
    url "https://files.pythonhosted.org/packages/b5/64/7660f8a4a8e53c924d0fa05dc3a55c9cee10bbd82b11c5afb27d44b096ce/markupsafe-3.0.3-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "c47a551199eb8eb2121d4f0f15ae0f923d31350ab9280078d1e5f12b249e0026"
  end

  resource "mpmath" do
    url "https://files.pythonhosted.org/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl"
    sha256 "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c"
  end

  resource "networkx" do
    url "https://files.pythonhosted.org/packages/9e/c9/b2622292ea83fbb4ec318f5b9ab867d0a28ab43c5717bb85b0a5f6b3b0a4/networkx-3.6.1-py3-none-any.whl"
    sha256 "d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762"
  end

  resource "packaging" do
    url "https://files.pythonhosted.org/packages/b7/b9/c538f279a4e237a006a2c98387d081e9eb060d203d8ed34467cc0f0b9b53/packaging-26.0-py3-none-any.whl"
    sha256 "b36f1fef9334a5588b4166f8bcd26a14e521f2b55e6b9de3aaa80d3ff7a37529"
  end

  resource "pillow" do
    url "https://files.pythonhosted.org/packages/53/4b/649056e4d22e1caa90816bf99cef0884aed607ed38075bd75f091a607a38/pillow-12.1.0-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "3413c2ae377550f5487991d444428f1a8ae92784aac79caa8b1e3b89b175f77e"
  end

  resource "opencv-python" do
    url "https://files.pythonhosted.org/packages/fc/6f/5a28fef4c4a382be06afe3938c64cc168223016fa520c5abaf37e8862aa5/opencv_python-4.13.0.92-cp37-abi3-macosx_13_0_arm64.whl"
    sha256 "caf60c071ec391ba51ed00a4a920f996d0b64e3e46068aac1f646b5de0326a19"
  end

  resource "pyyaml" do
    url "https://files.pythonhosted.org/packages/bd/9c/4d95bb87eb2063d20db7b60faa3840c1b18025517ae857371c4dd55a6b3a/pyyaml-6.0.3-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "34d5fcd24b8445fadc33f9cf348c1047101756fd760b4dacb5c3e99755703310"
  end

  resource "python-rapidjson" do
    url "https://files.pythonhosted.org/packages/18/5c/e33de01ce1476debf5c1acf9bfafcdcaeaef37834e3e6f723d15e1c481bb/python_rapidjson-1.23-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "3dcea6b19b3eee6a9dc5d8c1a20c828fa18d9d265893ce4c1858e8a85af944aa"
  end

  resource "requests" do
    url "https://files.pythonhosted.org/packages/1e/db/4254e3eabe8020b458f1a747140d32277ec7a271daf1d235b70dc0b4e6e3/requests-2.32.5-py3-none-any.whl"
    sha256 "2462f94637a34fd532264295e186976db0f5d453d1cdd31473c85a6a161affb6"
  end

  resource "regex" do
    url "https://files.pythonhosted.org/packages/f9/b6/921cc61982e538682bdf3bdf5b2c6ab6b34368da1f8e98a6c1ddc503c9cf/regex-2026.1.15-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "621f73a07595d83f28952d7bd1e91e9d1ed7625fb7af0064d3516674ec93a2a2"
  end

  resource "safetensors" do
    url "https://files.pythonhosted.org/packages/e8/00/374c0c068e30cd31f1e1b46b4b5738168ec79e7689ca82ee93ddfea05109/safetensors-0.7.0-cp38-abi3-macosx_11_0_arm64.whl"
    sha256 "94fd4858284736bb67a897a41608b5b0c2496c9bdb3bf2af1fa3409127f20d57"
  end

  resource "setuptools" do
    url "https://files.pythonhosted.org/packages/94/b8/f1f62a5e3c0ad2ff1d189590bfa4c46b4f3b6e49cef6f26c6ee4e575394d/setuptools-80.10.2-py3-none-any.whl"
    sha256 "95b30ddfb717250edb492926c92b5221f7ef3fbcc2b07579bcd4a27da21d0173"
  end

  resource "shellingham" do
    url "https://files.pythonhosted.org/packages/e0/f9/0595336914c5619e5f28a1fb793285925a8cd4b432c9da0a987836c7f822/shellingham-1.5.4-py2.py3-none-any.whl"
    sha256 "7ecfff8f2fd72616f7481040475a65b2bf8af90a56c89140852d1120324e8686"
  end

  resource "sympy" do
    url "https://files.pythonhosted.org/packages/a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl"
    sha256 "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5"
  end

  resource "tokenizers" do
    url "https://files.pythonhosted.org/packages/2e/47/174dca0502ef88b28f1c9e06b73ce33500eedfac7a7692108aec220464e7/tokenizers-0.22.2-cp39-abi3-macosx_11_0_arm64.whl"
    sha256 "1e418a55456beedca4621dbab65a318981467a2b188e982a23e117f115ce5001"
  end

  resource "torch" do
    url "https://files.pythonhosted.org/packages/4f/93/716b5ac0155f1be70ed81bacc21269c3ece8dba0c249b9994094110bfc51/torch-2.10.0-cp314-cp314-macosx_14_0_arm64.whl"
    sha256 "bf0d9ff448b0218e0433aeb198805192346c4fd659c852370d5cc245f602a06a"
  end

  resource "tqdm" do
    url "https://files.pythonhosted.org/packages/16/e1/3079a9ff9b8e11b846c6ac5c8b5bfb7ff225eee721825310c91b3b50304f/tqdm-4.67.3-py3-none-any.whl"
    sha256 "ee1e4c0e59148062281c49d80b25b67771a127c85fc9676d3be5f243206826bf"
  end

  resource "tritonclient" do
    url "https://files.pythonhosted.org/packages/15/89/75166c9b025bcdd0e86dc342ea590999be36078c2125d78b97a3c471db76/tritonclient-2.65.0-py3-none-any.whl"
    sha256 "cc3e58cf24a5f53fe24066e2037756560f6fdd2743a9d19fbb641b7580a63b4f"
  end

  resource "transformers" do
    url "https://files.pythonhosted.org/packages/71/d3/c16c3b3cf7655a67db1144da94b021c200ac1303f82428f2beef6c2e72bb/transformers-4.57.1-py3-none-any.whl"
    sha256 "b10d05da8fa67dc41644dbbf9bc45a44cb86ae33da6f9295f5fbf5b7890bd267"
  end

  resource "typer-slim" do
    url "https://files.pythonhosted.org/packages/c8/0a/4aca634faf693e33004796b6cee0ae2e1dba375a800c16ab8d3eff4bb800/typer_slim-0.21.1-py3-none-any.whl"
    sha256 "6e6c31047f171ac93cc5a973c9e617dbc5ab2bddc4d0a3135dc161b4e2020e0d"
  end

  resource "zope-event" do
    url "https://files.pythonhosted.org/packages/c2/b0/956902e5e1302f8c5d124e219c6bf214e2649f92ad5fce85b05c039a04c9/zope_event-6.1-py3-none-any.whl"
    sha256 "0ca78b6391b694272b23ec1335c0294cc471065ed10f7f606858fc54566c25a0"
  end

  resource "zope-interface" do
    url "https://files.pythonhosted.org/packages/00/7f/62d00ec53f0a6e5df0c984781e6f3999ed265129c4c3413df8128d1e0207/zope_interface-8.2-cp314-cp314-macosx_11_0_arm64.whl"
    sha256 "a87fc7517f825a97ff4a4ca4c8a950593c59e0f8e7bfe1b6f898a38d5ba9f9cf"
  end

  def install
    python = Formula["python@3.14"].opt_bin/"python3.14"
    venv = virtualenv_create(libexec, python, system_site_packages: true)

    resources.each do |r|
      r.stage do
        wheel = Dir["*.whl"].first
        target = wheel ? Pathname.pwd/wheel : Pathname.pwd
        system python, "-m", "pip", "--python=#{venv.root}/bin/python", "install",
               "--no-deps", "--ignore-installed", "--no-compile", "--only-binary=:all:", target
      end
    end

    system python, "-m", "pip", "--python=#{venv.root}/bin/python", "install",
           "--no-deps", "--ignore-installed", "--no-compile", buildpath
    (etc/"nuv-agent").mkpath

    env = {
      "DYLD_LIBRARY_PATH" => "#{HOMEBREW_PREFIX}/lib",
      "GI_TYPELIB_PATH" => "#{HOMEBREW_PREFIX}/lib/girepository-1.0",
      "GST_PLUGIN_PATH" => "#{HOMEBREW_PREFIX}/lib/gstreamer-1.0",
      "PYTHONNOUSERSITE" => "1",
      "PATH" => "#{HOMEBREW_PREFIX}/bin:#{HOMEBREW_PREFIX}/sbin:/usr/bin:/bin",
    }
    real_bin = libexec/"bin/nuv-agent"
    (bin/"nuv-agent").unlink if (bin/"nuv-agent").exist?
    (bin/"nuv-agent").write_env_script(real_bin, env)
  end

  def caveats
    <<~EOS
      Runtime bootstrap is enabled by default.
      On macOS, `nuv-agent setup` / `nuv-agent run` will try to:
      1) install Homebrew (if missing),
      2) install docker + colima (if missing),
      3) start a local Triton container when NUVION_ZSAD_BACKEND=triton.

      If Docker Desktop is already running, it is used first. Colima is only a fallback.
    EOS
  end

  service do
    run [opt_bin/"nuv-agent", "run"]
    keep_alive true
    environment_variables NUV_AGENT_CONFIG: etc/"nuv-agent/agent.env",
                          DYLD_LIBRARY_PATH: "#{HOMEBREW_PREFIX}/lib",
                          GI_TYPELIB_PATH: "#{HOMEBREW_PREFIX}/lib/girepository-1.0",
                          GST_PLUGIN_PATH: "#{HOMEBREW_PREFIX}/lib/gstreamer-1.0",
                          PYTHONNOUSERSITE: "1",
                          PATH: "#{HOMEBREW_PREFIX}/bin:#{HOMEBREW_PREFIX}/sbin:/usr/bin:/bin"
    log_path var/"log/nuv-agent.log"
    error_log_path var/"log/nuv-agent.log"
  end

  test do
    system bin/"nuv-agent", "config-path"
  end
end
