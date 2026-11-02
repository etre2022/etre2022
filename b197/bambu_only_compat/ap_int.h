// B197 research compatibility shim for *native C++ emulation only*.
// C++11-compatible variant exclusively for Bambu frontend exploration.
// Not an AMD/Xilinx header; does not model HLS scheduling, AXI, RTL or 4-state values.
// Only the ap_uint/ap_int features used by pinned public GDN kernel are implemented.
#ifndef B197_NATIVE_ONLY_AP_INT_SHIM_H_
#define B197_NATIVE_ONLY_AP_INT_SHIM_H_
#include <array>
#include <cstdint>
#include <cassert>
#include <type_traits>
namespace b197_native_ap {
template<int W> class ap_uint {
  static_assert(W > 0 && W % 64 == 0, "Only 64-bit-multiple widths are supported");
  static constexpr int N = W/64;
  std::array<std::uint64_t,N> x_{};
public:
  ap_uint() = default;
  template<class T, class = std::enable_if_t<std::is_integral<T>::value>>
  ap_uint(T v) { x_[0] = static_cast<std::uint64_t>(v); }
  const std::array<std::uint64_t,N>& words() const { return x_; }
  class slice {
    ap_uint& p_; int hi_,lo_;
  public:
    slice(ap_uint& p,int hi,int lo):p_(p),hi_(hi),lo_(lo){assert(lo>=0&&hi>=lo&&hi<W);}
    template <class T,class=std::enable_if_t<std::is_integral<T>::value>>
    slice& operator=(T v) {
      int n=hi_-lo_+1;
      assert(n<=64);int wi=lo_/64,bit=lo_%64;
      assert(bit+n<=64);
      std::uint64_t mask=n==64?~std::uint64_t(0):((std::uint64_t(1)<<n)-1);
      p_.x_[wi]=(p_.x_[wi]&~(mask<<bit))|((static_cast<std::uint64_t>(v)&mask)<<bit);
      return *this;
    }
    template<int OW> slice& operator=(const ap_uint<OW>& q) {
      int n=hi_-lo_+1;
      assert(n==OW&&lo_%64==0);
      for(int i=0;i<OW/64;i++) p_.x_[lo_/64+i]=q.words()[i];
      return *this;
    }
    unsigned to_uint() const {
      int n=hi_-lo_+1,bit=lo_%64;
      assert(n<=32&&bit+n<=64);
      std::uint64_t mask=(std::uint64_t(1)<<n)-1;
      return static_cast<unsigned>((p_.x_[lo_/64]>>bit)&mask);
    }
  };
  class const_slice {
    const ap_uint& p_;int hi_,lo_;
  public:
    const_slice(const ap_uint& p,int hi,int lo):p_(p),hi_(hi),lo_(lo){assert(lo>=0&&hi>=lo&&hi<W);}
    unsigned to_uint() const {
      int n=hi_-lo_+1,bit=lo_%64;
      assert(n<=32&&bit+n<=64);
      std::uint64_t mask=(std::uint64_t(1)<<n)-1;
      return static_cast<unsigned>((p_.x_[lo_/64]>>bit)&mask);
    }
  };
  slice range(int hi,int lo){return slice(*this,hi,lo);}
  const_slice range(int hi,int lo) const{return const_slice(*this,hi,lo);}
};
template<int W> class ap_int {
  static_assert(W==8,"Only ap_int<8> supported");
  int v_;
public:
  template<class Slice> ap_int(const Slice& s) {
    unsigned u=s.to_uint()&0xffu;
    v_=(u&0x80u)?static_cast<int>(u)-256:static_cast<int>(u);
  }
  int to_int() const{return v_;}
};
}
template<int W> using ap_uint=b197_native_ap::ap_uint<W>;
template<int W> using ap_int=b197_native_ap::ap_int<W>;
#endif
